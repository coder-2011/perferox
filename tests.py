"""High-signal unit tests for Perferox's host-owned contracts."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from perferox import db
from perferox.bench import BenchServingArgs, bench_serving_argv, parse_bench_serving_metrics
from perferox.process_host import MAIN_SESSION, _wait_for_main_event
from perferox.remote import RemoteResult, SessionRegistry
from perferox.semantic import DocumentIndex
from perferox.status import read_dashboard, read_trace_tail
from perferox.subagent import build_subagent_graph
from perferox.tools import cleanup_cloud_resources, provider_cli, sglang_bench_serving
from perferox.tui import request_end


@dataclass(slots=True)
class FakeRemoteSession:
  """Return one fixed remote command result without opening SSH."""

  session_id: str
  result: RemoteResult

  def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
    """Return the configured remote result."""
    return self.result

  def is_connected(self) -> bool:
    """Report the fake SSH session as live for host phase gates."""
    return True


class ToolBindingFakeModel(FakeMessagesListChatModel):
  """Let deterministic test messages pass through LangChain tool binding."""

  def bind_tools(self, tools: Any, **kwargs: Any) -> ToolBindingFakeModel:
    """Return this fake because its responses already contain tool calls."""
    return self


class DatabaseTestCase(unittest.TestCase):
  """Create one initialized temp SQLite database per test."""

  def setUp(self) -> None:
    """Open a fresh database."""
    self.tempdir = tempfile.TemporaryDirectory()
    self.db_path = Path(self.tempdir.name) / "perferox.sqlite"
    self.conn = db.connect(self.db_path)
    db.init_db(self.conn)

  def tearDown(self) -> None:
    """Close and delete the temp database."""
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

  def start_run(self, agent_id: int, command: str, *, commit: str = "a" * 40, intent: str | None = None, attempt_cap: int | None = None) -> int:
    """Start one canonical run without repeating identity boilerplate."""
    return db.start_benchmark_run(
      self.conn,
      agent_id=agent_id,
      repository="https://github.com/sgl-project/sglang.git",
      target_commit=commit,
      provider="runpod",
      resource_config={"gpu": "A100"},
      hardware_config="A100 80GB",
      server_config="tp=1 model=test",
      command=command,
      intent_key=intent or command,
      attempt_cap=attempt_cap,
    )


class BenchmarkContractTests(unittest.TestCase):
  """Protect benchmark command normalization and output parsing."""

  def test_serving_args_and_metrics_stay_stable(self) -> None:
    """Check the command/hash boundary and parsed metrics in one fixture."""
    args = BenchServingArgs(
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
    self.assertEqual(argv[argv.index("--extra-request-body") + 1], '{"mode":"stress","seed":7}')
    self.assertEqual(argv[argv.index("--header") + 1], "x-trace=perferox")
    self.assertNotIn("--timeout-s", argv)

    with self.assertRaises(ValidationError):
      BenchServingArgs(print_requests=True, backend="sglang")

    output = """
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
    self.assertEqual(metrics["request_rps"], 12.34)
    self.assertEqual(metrics["input_tps"], 1234.5)
    self.assertEqual(metrics["cache_hit_rate"], 0.75)
    self.assertEqual(metrics["error_rate"], 0.1)


class HostStateTests(DatabaseTestCase):
  """Protect deterministic SQLite-owned state transitions."""

  def test_existing_database_moves_intent_to_runs(self) -> None:
    """Migrate the old non-null experiment intent columns without losing data."""
    old_path = Path(self.tempdir.name) / "old.sqlite"
    with db.open_db(old_path) as conn:
      conn.executescript(
        """
        CREATE TABLE runs (
          agent_id INTEGER NOT NULL, run_id INTEGER NOT NULL, gpu TEXT NOT NULL DEFAULT '',
          started_at TEXT NOT NULL, finished_at TEXT, trace_ref TEXT, command TEXT NOT NULL DEFAULT '',
          exact_hash TEXT NOT NULL UNIQUE, error TEXT NOT NULL DEFAULT '', PRIMARY KEY(agent_id, run_id)
        );
        CREATE TABLE experiments (
          agent_id INTEGER NOT NULL, run_id INTEGER NOT NULL, intent_key TEXT NOT NULL,
          intent_embedding TEXT NOT NULL, request_rps REAL, input_tps REAL, output_tps REAL,
          ttft_p50_ms REAL, ttft_p99_ms REAL, tpot_p50_ms REAL, tpot_p99_ms REAL,
          error_rate REAL, cache_hit_rate REAL, peak_gpu_mem_gb REAL, startup_s REAL,
          warmup_s REAL, accept_length REAL, correctness_score REAL, PRIMARY KEY(agent_id, run_id)
        );
        INSERT INTO runs(agent_id, run_id, started_at, command, exact_hash) VALUES (0, 0, 'now', 'bench', 'hash');
        INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, request_rps) VALUES (0, 0, 'cache probe', '[1.0,0.0]', 4.5);
        """
      )
      db.init_db(conn)
      run = conn.execute("SELECT intent_key, intent_embedding FROM runs").fetchone()
      experiment = conn.execute("SELECT * FROM experiments").fetchone()
      columns = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
    self.assertEqual(dict(run), {"intent_key": "cache probe", "intent_embedding": "[1.0,0.0]"})
    self.assertEqual(experiment["request_rps"], 4.5)
    self.assertNotIn("intent_key", columns)

  def test_run_identity_repeat_caps_and_stop_are_host_owned(self) -> None:
    """Exercise complete identity, failed retries, cap counting, and stop."""
    with patch.object(db, "embed_intent", side_effect=([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0])):
      self.assertEqual(self.start_run(0, "bench a"), 0)
      db.log_experiment(self.conn, agent_id=0, run_id=0, metrics={"request_rps": 1})
      with self.assertRaisesRegex(ValueError, "exact experiment"):
        self.start_run(1, "bench a")
      self.assertEqual(self.start_run(1, "bench a", commit="b" * 40), 0)
      run_id = self.start_run(2, "fragile", intent="fragile retry", attempt_cap=1)
    db.mark_run_failed(self.conn, agent_id=2, run_id=run_id, error="remote crashed")
    with self.assertRaisesRegex(ValueError, "attempt cap reached"), patch.object(db, "embed_intent", return_value=[0.0, 1.0]):
      self.start_run(2, "second", attempt_cap=1)
    with patch.object(db, "embed_intent", return_value=[0.0, 1.0]):
      self.assertEqual(self.start_run(3, "fragile", intent="fragile retry"), 0)

    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-2", role="subagent", agent_id=2)
    self.assertEqual(db.request_soft_stop(self.conn), 2)
    with self.assertRaisesRegex(ValueError, "stop requested"), patch.object(db, "embed_intent", return_value=[0.0, 1.0]):
      self.start_run(2, "should not start")
    self.assertIn("remote crashed", self.run_row(agent_id=2)["error"])

  def test_subagent_reservations_are_serialized(self) -> None:
    """Reserve distinct agent ids across concurrent delegations."""
    def reserve() -> int:
      """Reserve through an independent process-style connection."""
      with db.open_db(self.db_path) as conn:
        return db.reserve_subagent(conn, active_cap=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
      agent_ids = sorted(pool.map(lambda _: reserve(), range(2)))

    self.assertEqual(agent_ids, [0, 1])
    self.assertEqual([row["status"] for row in self.conn.execute("SELECT status FROM agent_sessions ORDER BY agent_id")], ["starting", "starting"])
    with db.open_db(self.db_path) as conn, self.assertRaisesRegex(ValueError, "max active subagents reached"):
      db.reserve_subagent(conn, active_cap=2)
    self.assertEqual(db.request_soft_stop(self.conn), 2)

class ToolAndExperimentTests(DatabaseTestCase):
  """Exercise benchmark tools through fake SSH and real SQLite writes."""

  def test_benchmark_tool_marks_failure_and_returns_success_metrics(self) -> None:
    """Check started remote failure accounting and success metric output."""
    registry = SessionRegistry()
    db.register_cloud_resource(self.conn, agent_id=7, provider="runpod", resource_id="pod-7", environment={"gpu": "A100"})
    db.register_cloud_resource(self.conn, agent_id=8, provider="runpod", resource_id="pod-8", environment={"gpu": "A100"})
    registry.add(FakeRemoteSession("fail", RemoteResult(exit_status=2, stdout="", stderr="benchmark exploded")))
    fail_tool = sglang_bench_serving(registry, "fail", self.db_path, 7, "repo", "a" * 40, trace_ref="traces/agent-7.jsonl")
    with patch.object(db, "embed_intent", return_value=[1.0, 0.0]):
      failed = fail_tool.invoke({"intent_key": "failure smoke test", "hardware_config": "A100", "server_config": "tp=1", "num_prompts": 1, "timeout_s": 3.0})

    registry.add(
      FakeRemoteSession(
        "ok",
        RemoteResult(
          exit_status=0,
          stdout="Successful requests: 18\nRequest throughput (req/s): 12.34\nCache hit rate: 75.0%",
          stderr="",
        ),
      ),
    )
    ok_tool = sglang_bench_serving(registry, "ok", self.db_path, 8, "repo", "a" * 40)
    with patch.object(db, "embed_intent", return_value=[0.0, 1.0]):
      succeeded = ok_tool.invoke({"intent_key": "throughput smoke test", "hardware_config": "A100", "server_config": "tp=1", "num_prompts": 20})

    self.assertIn("run_id=0", failed)
    self.assertIn("exit_code=2", failed)
    self.assertIn("benchmark exploded", self.run_row(agent_id=7)["error"])
    self.assertIn('parsed_metrics={"cache_hit_rate":0.75,"error_rate":0.1,"request_rps":12.34}', succeeded)

  def test_experiment_logging_similarity_and_anomalies(self) -> None:
    """Check metric validation, normalization, similarity order, and anomalies."""
    with self.assertRaisesRegex(ValueError, "metrics must not be empty"):
      db.log_experiment(self.conn, agent_id=3, run_id=0)

    with patch.object(db, "embed_intent", side_effect=([1.0, 0.0], [0.0, 1.0], [0.9, 0.1])):
      first = self.start_run(3, "valid benchmark", intent="CUDA cache throughput")
      with self.assertRaisesRegex(ValueError, "unknown metric columns"):
        db.log_experiment(self.conn, agent_id=3, run_id=first, metrics={"made_up_metric": 1.0})
      db.log_experiment(self.conn, agent_id=3, run_id=first, metrics={"cache_hit_rate": 75, "error_rate": 25})
      second = self.start_run(3, "scheduler benchmark", intent="scheduler tail latency")
      db.log_experiment(self.conn, agent_id=3, run_id=second, metrics={"ttft_p99_ms": 10})
      matches = db.find_similar_experiments(self.conn, "cache-ish intent", limit=2)

    anomaly_id = db.log_anomaly(self.conn, agent_id=3, run_id=0, summary="cache-hit collapse on MI250")
    experiment = self.conn.execute("SELECT * FROM experiments WHERE agent_id = 3 AND run_id = 0").fetchone()
    anomaly = self.conn.execute("SELECT * FROM anomalies WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    self.assertEqual(experiment["cache_hit_rate"], 0.75)
    self.assertEqual(experiment["error_rate"], 0.25)
    self.assertEqual([match["intent_key"] for match in matches], ["CUDA cache throughput", "scheduler tail latency"])
    self.assertEqual(anomaly["summary"], "cache-hit collapse on MI250")

  def test_provider_resource_is_recorded_and_cleaned(self) -> None:
    """Prove paid resource identity survives tool execution and drives teardown."""
    tool = provider_cli("runpod", self.db_path, agent_id=4)
    outputs = ["exit_code=0\n{\"id\":\"pod-4\"}", "exit_code=0\ndeleted"]
    with patch("perferox.tools._run_argv", side_effect=outputs):
      created = tool.invoke({"arguments": ["pod", "create", "--gpu-id", "A100"]})
      errors = cleanup_cloud_resources(self.db_path, 4, "rpa_test")
    resource = self.conn.execute("SELECT * FROM cloud_resources WHERE agent_id = 4").fetchone()
    self.assertIn("resource_id=pod-4", created)
    self.assertEqual(errors, [])
    self.assertIsNotNone(resource["terminated_at"])

  def test_document_index_keeps_rows_aligned_with_vectors(self) -> None:
    """Rank cached documents without mutating their vector matrix."""
    with self.conn:
      self.conn.executemany(
        "INSERT INTO doc_chunks(source, chunk_id, text, embedding, updated_at) VALUES (?, ?, '', ?, '')",
        (("cache", "0", "[1.0,0.0]"), ("scheduler", "1", "[0.0,1.0]"), ("mixed", "2", "[0.8,0.2]")),
      )

    index = DocumentIndex.load(self.db_path)
    matches = index.search([1.0, 0.0], 2)

    self.assertEqual([document[0] for _, document in matches], ["cache", "mixed"])
    self.assertFalse(index.vectors.flags.writeable)

  def test_soft_stop_blocks_pending_provisioning_tool(self) -> None:
    """Route a stopped worker to summary without executing its requested tool."""
    calls = []

    @tool
    def provision() -> str:
      """Record whether a provisioning tool was incorrectly executed."""
      calls.append("provisioned")
      return "provisioned"

    model = ToolBindingFakeModel(responses=[
      AIMessage(content="", tool_calls=[{"name": "provision", "args": {}, "id": "call-1", "type": "tool_call"}]),
      AIMessage(content="stopped before provisioning"),
    ])
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-9", role="subagent", agent_id=9)
    db.request_soft_stop(self.conn)
    graph = build_subagent_graph(model, 9, SessionRegistry(), self.db_path, "repo", "commit", create_pod_tools=(provision,))

    result = graph.invoke({"agent_id": 9, "objective": "benchmark goal", "messages": []})

    self.assertEqual(calls, [])
    self.assertEqual(result["summary"], "stopped before provisioning")

  def test_cap_one_run_can_finalize_without_dangling_tool_call(self) -> None:
    """Run the reproduced cap boundary through benchmark, logging, and summary."""
    registry = SessionRegistry()
    registry.add(FakeRemoteSession("agent-10", RemoteResult(0, "Successful requests: 1\nRequest throughput (req/s): 2.5", "")))
    db.register_cloud_resource(self.conn, agent_id=10, provider="runpod", resource_id="pod-10", environment={"gpu": "A100"})
    benchmark_args = {
      "intent_key": "single request cap boundary",
      "hardware_config": "A100",
      "server_config": "tp=1",
      "num_prompts": 1,
    }
    model = ToolBindingFakeModel(responses=[
      AIMessage(content="connected"),
      AIMessage(content="setup_ready: exact commit"),
      AIMessage(content="", tool_calls=[{"name": "sglang_bench_serving", "args": benchmark_args, "id": "bench-1", "type": "tool_call"}]),
      AIMessage(content="", tool_calls=[{"name": "log_experiment", "args": {"run_id": 0, "metrics": {"request_rps": 2.5, "error_rate": 0}}, "id": "log-1", "type": "tool_call"}]),
      AIMessage(content="benchmark complete"),
      AIMessage(content="run 0 completed and logged"),
    ])
    graph = build_subagent_graph(model, 10, registry, self.db_path, "repo", "a" * 40, attempt_cap=1)
    with patch.object(db, "embed_intent", return_value=[1.0, 0.0]):
      result = graph.invoke({"agent_id": 10, "objective": "cap boundary", "messages": []})

    self.assertEqual(result["summary"], "run 0 completed and logged")
    self.assertIsNotNone(self.run_row(10)["finished_at"])
    self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM experiments WHERE agent_id = 10").fetchone()[0], 1)
    tool_ids = {message.tool_call_id for message in result["messages"] if message.type == "tool"}
    self.assertEqual(tool_ids, {"bench-1", "log-1"})


class TUIWiringTests(DatabaseTestCase):
  """Protect the TUI bridge without model, browser, SSH, or cloud work."""

  def test_dashboard_trace_tail_and_soft_stop_flow(self) -> None:
    """Read live state, preserve notifications, then request soft stop."""
    trace_path = Path(self.tempdir.name) / "main.jsonl"
    lines = [json.dumps({"payload": {"main": {"messages": [{"content": f"cache pressure {index}"}]}}}, separators=(",", ":")) for index in range(30)]
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path))
    db.record_agent_session(self.conn, session_name="perferox-agent-0", role="subagent", agent_id=0, trace_ref=str(trace_path))
    db.append_explorer_state(self.conn, agent_id=None, line="explorer saw cache pressure")
    with patch.object(db, "embed_intent", return_value=[1.0, 0.0]):
      self.start_run(0, "bench cache", intent="cache pressure")
    db.log_anomaly(self.conn, agent_id=0, run_id=0, summary="cache pressure anomaly")

    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
      snapshot = read_dashboard(self.db_path)
    delivered = self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"]
    claimed = db.claim_main_notifications(self.conn)
    with patch("perferox.process_host.shutil.which", return_value=True), patch("perferox.process_host.subprocess.run") as run:
      run.side_effect = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 1)]
      stopped = request_end(self.db_path)
      update, notification_ids = _wait_for_main_event(self.db_path, poll_s=0)

    trace_text = "\n".join(snapshot.trace_lines)
    tail_lines = read_trace_tail([str(trace_path)], 5)
    subagent = next(session for session in snapshot.sessions if session["session_name"] == "perferox-agent-0")
    self.assertEqual(snapshot.main_status, "running")
    self.assertEqual(snapshot.runs, 1)
    self.assertEqual(snapshot.running_runs, 1)
    self.assertEqual(snapshot.anomaly_count, 1)
    self.assertEqual(snapshot.recent_runs[0]["label"], "cache pressure")
    self.assertEqual(subagent["run_count"], 1)
    self.assertEqual(snapshot.anomalies[0]["summary"], "cache pressure anomaly")
    self.assertIn("cache pressure 29", trace_text)
    self.assertIn("explorer saw cache pressure", trace_text)
    self.assertIsNone(delivered)
    self.assertTrue(claimed)
    self.assertIn("cache pressure 25", tail_lines[0])
    self.assertIn("cache pressure 29", tail_lines[-1])
    self.assertEqual(stopped, 2)
    self.assertIsNone(update)
    self.assertEqual(notification_ids, [])

  def test_installed_process_host_runs_outside_checkout(self) -> None:
    """Use the active interpreter to launch the installed module from a temp cwd."""
    result = subprocess.run(
      [sys.executable, "-m", "perferox.process_host", "--help"],
      cwd=self.tempdir.name,
      text=True,
      capture_output=True,
      check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("launch-main", result.stdout)


if __name__ == "__main__":
  unittest.main()
