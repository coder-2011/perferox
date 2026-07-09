"""High-signal unit tests for Perferox's host-owned contracts."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from perferox import cli, db
from perferox.agent_runner import MAIN_SESSION, _wait_for_main_event
from perferox.bench import BenchServingArgs, bench_serving_argv, parse_bench_serving_metrics
from perferox.remote import RemoteResult, SessionRegistry
from perferox.tools import sglang_bench_serving
from perferox.tui import launch_main, read_dashboard, read_trace_tail, request_end


@dataclass(slots=True)
class FakeRemoteSession:
  """Return one fixed remote command result without opening SSH."""

  session_id: str
  result: RemoteResult

  def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
    """Return the configured remote result."""
    return self.result


class DatabaseTestCase(unittest.TestCase):
  """Create one initialized temp SQLite database per test."""

  def setUp(self) -> None:
    self.tempdir = tempfile.TemporaryDirectory()
    self.db_path = Path(self.tempdir.name) / "perferox.sqlite"
    self.conn = db.connect(self.db_path)
    db.init_db(self.conn)

  def tearDown(self) -> None:
    self.conn.close()
    self.tempdir.cleanup()

  def run_row(self, agent_id: int, run_id: int = 0) -> sqlite3.Row:
    """Fetch one run row by its deterministic host-owned key."""
    row = self.conn.execute(
      "SELECT * FROM runs WHERE agent_id = ? AND run_id = ?",
      (agent_id, run_id),
    ).fetchone()
    self.assertIsNotNone(row)
    return row


class BenchmarkCommandTests(unittest.TestCase):
  """Protect the benchmark command shape that feeds exact hashes."""

  def test_bench_serving_argv_keeps_stable_flags_and_compact_json(self) -> None:
    args = BenchServingArgs(
      output_details=True,
      cache_report=True,
      num_prompts=8,
      request_rate=2.5,
      extra_request_body={"mode": "stress", "seed": 7},
      header={"x-trace": "perferox"},
      timeout_s=12.0,
    )

    argv = bench_serving_argv(args)

    self.assertEqual(argv[:3], ["python", "-m", "sglang.benchmark.serving"])
    self.assertIn("--output-details", argv)
    self.assertIn("--cache-report", argv)
    self.assertEqual(argv[argv.index("--num-prompts") + 1], "8")
    self.assertEqual(argv[argv.index("--request-rate") + 1], "2.5")
    self.assertEqual(argv[argv.index("--extra-request-body") + 1], '{"mode":"stress","seed":7}')
    self.assertEqual(argv[argv.index("--header") + 1], "x-trace=perferox")
    self.assertNotIn("--timeout-s", argv)

  def test_bench_serving_args_reject_invalid_cli_combinations(self) -> None:
    with self.assertRaises(ValidationError):
      BenchServingArgs(print_requests=True, backend="sglang")

  def test_parse_bench_serving_metrics_matches_log_experiment_columns(self) -> None:
    output = """
    exit_code=0
    Successful requests:                     18
    Request throughput (req/s):             12.34
    Input token throughput (tok/s):         1234.50
    Output token throughput (tok/s):        456.70
    Median TTFT (ms):                       45.67
    P99 TTFT (ms):                          123.45
    Median TPOT (ms):                       5.60
    P99 TPOT (ms):                          9.80
    Accept length:                          3.25
    Cache hit rate:                         75.0%
    """

    metrics = parse_bench_serving_metrics(output, expected_requests=20)

    self.assertEqual(
      metrics,
      {
        "request_rps": 12.34,
        "input_tps": 1234.5,
        "output_tps": 456.7,
        "ttft_p50_ms": 45.67,
        "ttft_p99_ms": 123.45,
        "tpot_p50_ms": 5.6,
        "tpot_p99_ms": 9.8,
        "accept_length": 3.25,
        "cache_hit_rate": 0.75,
        "error_rate": 0.1,
      },
    )


class RunLifecycleTests(DatabaseTestCase):
  """Protect deterministic run IDs, duplicate hashes, and cap behavior."""

  def test_start_benchmark_run_assigns_agent_local_run_ids(self) -> None:
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench a"), 0)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench b"), 1)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=1, command="bench c"), 0)

    rows = self.conn.execute("SELECT agent_id, run_id FROM runs ORDER BY agent_id, run_id").fetchall()
    self.assertEqual([(row["agent_id"], row["run_id"]) for row in rows], [(0, 0), (0, 1), (1, 0)])

  def test_duplicate_exact_command_is_rejected(self) -> None:
    db.start_benchmark_run(self.conn, agent_id=0, command="same benchmark")

    with self.assertRaises(sqlite3.IntegrityError):
      db.start_benchmark_run(self.conn, agent_id=1, command="same benchmark")

  def test_attempt_cap_counts_started_failed_runs(self) -> None:
    run_id = db.start_benchmark_run(self.conn, agent_id=2, command="fragile benchmark", attempt_cap=1)
    db.mark_run_failed(self.conn, agent_id=2, run_id=run_id, error="remote crashed")

    with self.assertRaisesRegex(ValueError, "attempt cap reached"):
      db.start_benchmark_run(self.conn, agent_id=2, command="second benchmark", attempt_cap=1)

    row = self.run_row(agent_id=2)
    self.assertIn("remote crashed", row["error"])
    self.assertIsNotNone(row["finished_at"])

  def test_soft_stop_blocks_new_run_ids_inside_db_transaction(self) -> None:
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-2", role="subagent", agent_id=2)

    self.assertEqual(db.request_soft_stop(self.conn), 2)

    with self.assertRaisesRegex(ValueError, "stop requested"):
      db.start_benchmark_run(self.conn, agent_id=2, command="should not start")
    rows = self.conn.execute("SELECT * FROM runs").fetchall()
    self.assertEqual(rows, [])


class BenchmarkToolE2ETests(DatabaseTestCase):
  """Exercise the benchmark tool through fake SSH and real SQLite writes."""

  def test_remote_failure_starts_run_and_marks_it_failed(self) -> None:
    session_id = "agent-7"
    registry = SessionRegistry()
    result = RemoteResult(exit_status=2, stdout="", stderr="benchmark exploded")
    registry.add(FakeRemoteSession(session_id, result))
    tool = sglang_bench_serving(registry, session_id, self.db_path, agent_id=7, trace_ref="traces/agent-7.jsonl")

    result = tool.invoke({"output_details": True, "cache_report": True, "num_prompts": 1, "timeout_s": 3.0})

    run = self.run_row(agent_id=7)
    notifications = db.take_main_notifications(self.conn)
    self.assertIn("run_id=0", result)
    self.assertIn("exit_code=2", result)
    self.assertIn("benchmark exploded", run["error"])
    self.assertIsNotNone(run["finished_at"])
    self.assertIn("python -m sglang.benchmark.serving", run["command"])
    self.assertEqual([row["kind"] for row in notifications], ["run_started", "run_failed"])

  def test_remote_success_returns_parsed_metrics(self) -> None:
    session_id = "agent-8"
    registry = SessionRegistry()
    registry.add(
      FakeRemoteSession(
        session_id,
        RemoteResult(
          exit_status=0,
          stdout="""
          Successful requests:                     18
          Request throughput (req/s):             12.34
          Cache hit rate:                         75.0%
          """,
          stderr="",
        ),
      ),
    )
    tool = sglang_bench_serving(registry, session_id, self.db_path, agent_id=8)

    result = tool.invoke({"num_prompts": 20})

    self.assertIn("run_id=0", result)
    self.assertIn('parsed_metrics={"cache_hit_rate":0.75,"error_rate":0.1,"request_rps":12.34}', result)

  def test_stop_requested_refuses_benchmark_before_remote_run(self) -> None:
    session_id = "agent-9"
    registry = SessionRegistry()
    registry.add(FakeRemoteSession(session_id, RemoteResult(exit_status=0, stdout="unused", stderr="")))
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-9", role="subagent", agent_id=9)
    db.request_soft_stop(self.conn)
    tool = sglang_bench_serving(registry, session_id, self.db_path, agent_id=9)

    result = tool.invoke({"num_prompts": 1})

    self.assertIn("benchmark not started: ValueError: stop requested; wrap up", result)
    rows = self.conn.execute("SELECT * FROM runs").fetchall()
    self.assertEqual(rows, [])


class ExperimentLoggingTests(DatabaseTestCase):
  """Protect the narrow path from started benchmark run to logged experiment."""

  def test_log_experiment_requires_started_run_and_rejects_unknown_metrics(self) -> None:
    with self.assertRaisesRegex(ValueError, "no unfinished successful benchmark run"):
      db.log_experiment(self.conn, agent_id=3, intent_key="no run")

    db.start_benchmark_run(self.conn, agent_id=3, command="valid benchmark")
    with self.assertRaisesRegex(ValueError, "unknown metric columns"):
      db.log_experiment(self.conn, agent_id=3, intent_key="bad metric", metrics={"made_up_metric": 1.0})

    with patch.object(db, "embed_intent", return_value=[1.0, 0.0]):
      run_id = db.log_experiment(
        self.conn,
        agent_id=3,
        intent_key="SGLang CUDA radix cache",
        metrics={"request_rps": 12.5, "cache_hit_rate": 75, "error_rate": 25},
      )

    self.assertEqual(run_id, 0)
    run = self.run_row(agent_id=3)
    experiment = self.conn.execute("SELECT * FROM experiments WHERE agent_id = 3 AND run_id = 0").fetchone()
    self.assertIsNotNone(run["finished_at"])
    self.assertEqual(experiment["intent_embedding"], "[1.0,0.0]")
    self.assertEqual(experiment["request_rps"], 12.5)
    self.assertEqual(experiment["cache_hit_rate"], 0.75)
    self.assertEqual(experiment["error_rate"], 0.25)

  def test_find_similar_experiments_orders_by_embedding_score(self) -> None:
    with patch.object(db, "embed_intent", side_effect=([1.0, 0.0], [0.0, 1.0], [0.9, 0.1])):
      db.start_benchmark_run(self.conn, agent_id=4, command="cache benchmark")
      db.log_experiment(self.conn, agent_id=4, intent_key="CUDA cache throughput")
      db.start_benchmark_run(self.conn, agent_id=4, command="scheduler benchmark")
      db.log_experiment(self.conn, agent_id=4, intent_key="scheduler tail latency")
      matches = db.find_similar_experiments(self.conn, "cache-ish intent", limit=2)

    self.assertEqual([match["intent_key"] for match in matches], ["CUDA cache throughput", "scheduler tail latency"])
    self.assertGreater(matches[0]["score"], matches[1]["score"])

  def test_log_anomaly_writes_summary_for_existing_run(self) -> None:
    db.start_benchmark_run(self.conn, agent_id=5, command="weird benchmark")
    anomaly_id = db.log_anomaly(self.conn, agent_id=5, run_id=0, summary="cache-hit collapse on MI250")

    anomaly = self.conn.execute("SELECT * FROM anomalies WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    self.assertEqual(anomaly["summary"], "cache-hit collapse on MI250")
    self.assertEqual(anomaly["agent_id"], 5)
    self.assertEqual(anomaly["run_id"], 0)


class AgentSessionTests(DatabaseTestCase):
  """Protect tmux-style agent session lifecycle rows."""

  def test_agent_session_can_be_recorded_ended_and_reused(self) -> None:
    db.record_agent_session(self.conn, session_name="perferox-main", role="main", agent_id=0, trace_ref="traces/main.jsonl")
    running = self.conn.execute("SELECT * FROM agent_sessions WHERE session_name = 'perferox-main'").fetchone()
    self.assertEqual(running["status"], "running")

    self.assertTrue(db.request_agent_end(self.conn, session_name="perferox-main"))
    self.assertTrue(db.finish_agent_session(self.conn, session_name="perferox-main", status="done"))
    self.assertFalse(db.finish_agent_session(self.conn, session_name="perferox-main", status="done"))

    db.record_agent_session(self.conn, session_name="perferox-main", role="main", agent_id=0)
    reused = self.conn.execute("SELECT * FROM agent_sessions WHERE session_name = 'perferox-main'").fetchone()
    self.assertEqual(reused["status"], "running")


class MainNotificationTests(DatabaseTestCase):
  """Protect the read-once notification queue for main-agent wakeups."""

  def test_written_rows_notify_main_once(self) -> None:
    db.start_benchmark_run(self.conn, agent_id=6, command="notify benchmark")
    db.mark_run_failed(self.conn, agent_id=6, run_id=0, error="timeout")

    first_batch = db.take_main_notifications(self.conn)
    second_batch = db.take_main_notifications(self.conn)

    self.assertEqual([row["kind"] for row in first_batch], ["run_started", "run_failed"])
    self.assertEqual(second_batch, [])
    delivered = self.conn.execute("SELECT COUNT(*) FROM main_notifications WHERE delivered_at IS NOT NULL").fetchone()[0]
    self.assertEqual(delivered, 2)


class TUIWiringTests(DatabaseTestCase):
  """Protect the live TUI bridge without launching model, browser, SSH, or cloud work."""

  def test_dashboard_reads_live_db_and_trace_without_consuming_notifications(self) -> None:
    trace_path = Path(self.tempdir.name) / "main.jsonl"
    lines = [json.dumps({"payload": {"main": {"messages": [{"content": f"cache pressure {index}"}]}}}, separators=(",", ":")) for index in range(30)]
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path))
    db.record_agent_session(self.conn, session_name="perferox-agent-0", role="subagent", agent_id=0, trace_ref=str(trace_path))
    db.append_explorer_state(self.conn, agent_id=None, line="explorer saw cache pressure")
    db.start_benchmark_run(self.conn, agent_id=0, command="bench cache")
    db.log_anomaly(self.conn, agent_id=0, run_id=0, summary="cache pressure anomaly")

    snapshot = read_dashboard(self.db_path)

    trace_text = "\n".join(snapshot.trace_lines)
    delivered = self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"]
    subagent_session = next(session for session in snapshot.sessions if session["session_name"] == "perferox-agent-0")
    self.assertEqual(snapshot.main_status, "running")
    self.assertEqual(snapshot.runs, 1)
    self.assertEqual(subagent_session["run_count"], 1)
    self.assertEqual(snapshot.anomalies[0]["summary"], "cache pressure anomaly")
    self.assertIn("cache pressure 29", trace_text)
    self.assertIn("explorer saw cache pressure", trace_text)
    self.assertIn("sqlite run_started", trace_text)
    self.assertIsNone(delivered)
    tail_lines = read_trace_tail([str(trace_path)], 5)
    self.assertIn("cache pressure 25", tail_lines[0])
    self.assertIn("cache pressure 29", tail_lines[-1])

  def test_end_request_reaches_runner_soft_stop_state(self) -> None:
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-1", role="subagent", agent_id=1)

    with patch("perferox.agent_runner.shutil.which", return_value=True), patch("perferox.agent_runner.subprocess.run") as run:
      run.return_value.returncode = 0
      stopped = request_end(self.db_path)
      update = _wait_for_main_event(self.db_path, poll_s=0)

    statuses = {
      row["session_name"]: row["status"]
      for row in self.conn.execute("SELECT session_name, status FROM agent_sessions").fetchall()
    }
    self.assertEqual(stopped, 2)
    self.assertEqual(statuses[MAIN_SESSION], "ending")
    self.assertEqual(statuses["perferox-agent-1"], "ending")
    self.assertIn("End requested", update)

    self.assertTrue(db.finish_agent_session(self.conn, session_name="perferox-agent-1", status="exited"))
    self.assertIsNone(_wait_for_main_event(self.db_path, poll_s=0))

  def test_launch_main_uses_existing_runner_cli(self) -> None:
    with patch("perferox.tui.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="started", stderr="")) as run:
      result = launch_main(Path(self.tempdir.name), self.db_path, Path(self.tempdir.name) / "traces", "stress SGLang")

    command = run.call_args.args[0]
    self.assertEqual(result.stdout, "started")
    self.assertEqual(command[:5], ["uv", "run", "python", "-m", "perferox.agent_runner"])
    self.assertIn("launch-main", command)
    self.assertIn("--objective", command)
    self.assertEqual(command[command.index("--objective") + 1], "stress SGLang")


class CLITests(DatabaseTestCase):
  """Protect the no-TUI soft stop command."""

  def test_end_command_requests_soft_stop(self) -> None:
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")

    with patch("builtins.print"):
      result = cli.main(["--cwd", self.tempdir.name, "--db-path", str(self.db_path), "end"])

    status = self.conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()["status"]
    self.assertEqual(result, 0)
    self.assertEqual(status, "ending")
    self.assertIn("soft stop requested from CLI", db.read_explorer_state(self.conn))


if __name__ == "__main__":
  unittest.main()
