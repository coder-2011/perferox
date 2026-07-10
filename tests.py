"""High-signal unit tests for Perferox's host-owned contracts."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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
from perferox.status import read_dashboard, read_trace_tail, refresh_sessions
from perferox.subagent import build_subagent_graph
from perferox.tools import sglang_bench_serving
from perferox.tui import request_end


@dataclass(slots=True)
class FakeRemoteSession:
  """Return one fixed remote command result without opening SSH."""

  session_id: str
  result: RemoteResult

  def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
    """Return the configured remote result."""
    return self.result


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

  def test_run_ids_hash_caps_and_stop_are_host_owned(self) -> None:
    """Exercise run assignment, duplicate rejection, cap counting, and stop."""
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench a"), 0)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench b"), 1)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=1, command="bench c"), 0)
    with self.assertRaises(sqlite3.IntegrityError):
      db.start_benchmark_run(self.conn, agent_id=1, command="bench a")

    run_id = db.start_benchmark_run(self.conn, agent_id=2, command="fragile", attempt_cap=1)
    db.mark_run_failed(self.conn, agent_id=2, run_id=run_id, error="remote crashed")
    with self.assertRaisesRegex(ValueError, "attempt cap reached"):
      db.start_benchmark_run(self.conn, agent_id=2, command="second", attempt_cap=1)

    def reserve(_: int) -> int:
      """Reserve through an independent process-style connection."""
      with closing(db.connect(self.db_path)) as conn:
        return db.reserve_subagent(conn, active_cap=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
      self.assertEqual(sorted(pool.map(reserve, range(2))), [3, 4])
    with self.assertRaisesRegex(ValueError, "max active subagents reached"):
      db.reserve_subagent(self.conn, active_cap=2)
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main")
    db.record_agent_session(self.conn, session_name="perferox-agent-2", role="subagent", agent_id=2)
    self.assertEqual(db.request_soft_stop(self.conn), 4)
    with self.assertRaisesRegex(ValueError, "stop requested"):
      db.start_benchmark_run(self.conn, agent_id=2, command="should not start")
    self.assertIn("remote crashed", self.run_row(agent_id=2)["error"])

  def test_refresh_preserves_sessions_registered_after_its_snapshot(self) -> None:
    """Keep a newly registered worker live and allow a missing worker to recover."""
    db.record_agent_session(self.conn, session_name="old", role="subagent", agent_id=0, trace_ref="old.jsonl")

    def probe(*args, **kwargs):
      """Register a worker after refresh selected its candidate rows."""
      db.record_agent_session(self.conn, session_name="new", role="subagent", agent_id=1, trace_ref="new.jsonl")
      return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", side_effect=probe):
      missing = refresh_sessions(self.conn)
    db.record_agent_session(self.conn, session_name="old", role="subagent", agent_id=0, trace_ref="old.jsonl")
    self.assertEqual(missing, ["agent-0 tmux missing; trace old.jsonl"])
    self.assertEqual(dict(self.conn.execute("SELECT session_name, status FROM agent_sessions")), {"old": "running", "new": "running"})

class ToolAndExperimentTests(DatabaseTestCase):
  """Exercise benchmark tools through fake SSH and real SQLite writes."""

  def test_benchmark_tool_marks_failure_and_returns_success_metrics(self) -> None:
    """Check started remote failure accounting and success metric output."""
    registry = SessionRegistry()
    registry.add(FakeRemoteSession("fail", RemoteResult(exit_status=2, stdout="", stderr="benchmark exploded")))
    fail_tool = sglang_bench_serving(registry, "fail", self.db_path, agent_id=7, trace_ref="traces/agent-7.jsonl")
    failed = fail_tool.invoke({"output_details": True, "cache_report": True, "num_prompts": 1, "timeout_s": 3.0})

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
    ok_tool = sglang_bench_serving(registry, "ok", self.db_path, agent_id=8)
    succeeded = ok_tool.invoke({"num_prompts": 20})

    self.assertIn("run_id=0", failed)
    self.assertIn("exit_code=2", failed)
    self.assertIn("benchmark exploded", self.run_row(agent_id=7)["error"])
    self.assertIn('parsed_metrics={"cache_hit_rate":0.75,"error_rate":0.1,"request_rps":12.34}', succeeded)

  def test_experiment_logging_similarity_and_anomalies(self) -> None:
    """Check metric validation, normalization, similarity order, and anomalies."""
    with self.assertRaisesRegex(ValueError, "no unfinished successful benchmark run"):
      db.log_experiment(self.conn, agent_id=3, intent_key="no run")

    db.start_benchmark_run(self.conn, agent_id=3, command="valid benchmark")
    with self.assertRaisesRegex(ValueError, "unknown metric columns"):
      db.log_experiment(self.conn, agent_id=3, intent_key="bad metric", metrics={"made_up_metric": 1.0})

    with patch.object(db, "embed_intent", side_effect=([1.0, 0.0], [0.0, 1.0], [0.9, 0.1])):
      db.log_experiment(self.conn, agent_id=3, intent_key="CUDA cache throughput", metrics={"cache_hit_rate": 75, "error_rate": 25})
      db.start_benchmark_run(self.conn, agent_id=3, command="scheduler benchmark")
      db.log_experiment(self.conn, agent_id=3, intent_key="scheduler tail latency")
      matches = db.find_similar_experiments(self.conn, "cache-ish intent", limit=2)

    anomaly_id = db.log_anomaly(self.conn, agent_id=3, run_id=0, summary="cache-hit collapse on MI250")
    experiment = self.conn.execute("SELECT * FROM experiments WHERE agent_id = 3 AND run_id = 0").fetchone()
    anomaly = self.conn.execute("SELECT * FROM anomalies WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    self.assertEqual(experiment["intent_embedding"], "[1.0,0.0]")
    self.assertEqual(experiment["cache_hit_rate"], 0.75)
    self.assertEqual(experiment["error_rate"], 0.25)
    self.assertEqual([match["intent_key"] for match in matches], ["CUDA cache throughput", "scheduler tail latency"])
    self.assertEqual(anomaly["summary"], "cache-hit collapse on MI250")

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
    db.start_benchmark_run(self.conn, agent_id=0, command="bench cache")
    db.log_anomaly(self.conn, agent_id=0, run_id=0, summary="cache pressure anomaly")

    snapshot = read_dashboard(self.db_path)
    delivered = self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"]
    db.take_main_notifications(self.conn)
    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=f"{MAIN_SESSION}\n", stderr="")):
      stopped = request_end(self.db_path)
      update = _wait_for_main_event(self.db_path, poll_s=0)

    trace_text = "\n".join(snapshot.trace_lines)
    tail_lines = read_trace_tail([str(trace_path)], 5)
    subagent = next(session for session in snapshot.sessions if session["session_name"] == "perferox-agent-0")
    self.assertEqual(snapshot.main_status, "running")
    self.assertEqual(snapshot.runs, 1)
    self.assertEqual(snapshot.running_runs, 1)
    self.assertEqual(snapshot.anomaly_count, 1)
    self.assertEqual(snapshot.recent_runs[0]["label"], "bench cache")
    self.assertEqual(subagent["run_count"], 1)
    self.assertEqual(snapshot.anomalies[0]["summary"], "cache pressure anomaly")
    self.assertIn("cache pressure 29", trace_text)
    self.assertIn("explorer saw cache pressure", trace_text)
    self.assertIsNone(delivered)
    self.assertIn("cache pressure 25", tail_lines[0])
    self.assertIn("cache pressure 29", tail_lines[-1])
    self.assertEqual(stopped, 2)
    self.assertIsNone(update)


if __name__ == "__main__":
  unittest.main()
