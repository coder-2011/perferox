"""High-signal unit tests for Perferox's host-owned contracts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from lambda_labs import request as lambda_request
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from perferox import auth as perferox_auth
from perferox import db, providers, xai_oauth
from perferox.auth import cloud_environment, modal_cloud_key
from perferox.bench import BenchServingArgs, bench_serving_argv, parse_bench_serving_metrics
from perferox.process_host import MAIN_SESSION, _wait_for_main_event
from perferox.remote import RemoteResult, SessionRegistry
from perferox.status import read_dashboard, read_trace_tail, refresh_sessions
from perferox.subagent import build_subagent_graph
from perferox.tools import (
  MODAL_CPU_CORES,
  MODAL_MEMORY_MIB,
  MODAL_READY_TIMEOUT_S,
  cleanup_cloud_resource,
  create_modal_sandbox,
  provider_cli,
  sglang_bench_serving,
)
from perferox.tui import request_end


@dataclass(slots=True)
class FakeRemoteSession:
  """Return one fixed remote command result without opening SSH."""

  session_id: str
  result: RemoteResult
  commands: list[str] = field(default_factory=list)

  def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
    """Return the configured remote result."""
    self.commands.append(command)
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
      gpu="H100 SXM 80GB x1",
      server_command="python -m sglang.launch_server --model model-a",
      model_state="model-a@revision-1",
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
    self.assertNotIn("--server-command", argv)

    with self.assertRaises(ValidationError):
      BenchServingArgs(
        gpu=args.gpu,
        server_command=args.server_command,
        model_state=args.model_state,
        print_requests=True,
        backend="sglang",
      )

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


class LambdaLabsTests(unittest.TestCase):
  """Protect the Lambda API request contract used by the bundled CLI."""

  def test_request_uses_documented_headers_and_json(self) -> None:
    """Send the API-key bearer header, JSON accept header, and compact body."""
    with patch.dict(os.environ, {"LAMBDA_API_KEY": "test-key"}), patch("lambda_labs.urlopen") as urlopen:
      response = urlopen.return_value.__enter__.return_value
      response.read.return_value = b'{"data":{"instance_ids":["instance-1"]}}'
      payload = lambda_request("POST", "instance-operations/launch", {"quantity": 1})

    request = urlopen.call_args.args[0]
    self.assertEqual(payload, {"instance_ids": ["instance-1"]})
    self.assertEqual(request.get_full_url(), "https://cloud.lambda.ai/api/v1/instance-operations/launch")
    self.assertEqual(request.get_header("Accept"), "application/json")
    self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
    self.assertEqual(request.data, b'{"quantity":1}')


class HostStateTests(DatabaseTestCase):
  """Protect deterministic SQLite-owned state transitions."""

  def test_run_ids_hash_caps_and_stop_are_host_owned(self) -> None:
    """Exercise run assignment, duplicate rejection, cap counting, and stop."""
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench a"), 0)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=0, command="bench b"), 1)
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=1, command="bench c"), 0)
    with self.assertRaises(sqlite3.IntegrityError):
      db.start_benchmark_run(self.conn, agent_id=1, command="bench a")
    self.assertEqual(db.start_benchmark_run(self.conn, agent_id=1, command="bench a", commit="different-commit"), 1)

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
    """Keep workers registered after the list snapshot live."""
    db.record_agent_session(self.conn, session_name="old", role="subagent", agent_id=0, trace_ref="old.jsonl")

    def probe(*args, **kwargs):
      """Register a worker after refresh selected its candidate rows."""
      db.record_agent_session(self.conn, session_name="old", role="subagent", agent_id=0, trace_ref="old.jsonl")
      db.record_agent_session(self.conn, session_name="new", role="subagent", agent_id=1, trace_ref="new.jsonl")
      return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", side_effect=probe):
      missing = refresh_sessions(self.conn)
    self.assertEqual(missing, [])
    self.assertEqual(dict(self.conn.execute("SELECT session_name, status FROM agent_sessions")), {"old": "running", "new": "running"})

class ToolAndExperimentTests(DatabaseTestCase):
  """Exercise benchmark tools through fake SSH and real SQLite writes."""

  def test_benchmark_tool_marks_failure_and_returns_success_metrics(self) -> None:
    """Check started remote failure accounting and success metric output."""
    registry = SessionRegistry()
    registry.add(FakeRemoteSession("fail", RemoteResult(exit_status=2, stdout="", stderr="benchmark exploded")))
    fail_tool = sglang_bench_serving(registry, "fail", self.db_path, agent_id=7, trace_ref="traces/agent-7.jsonl")
    identity = {"gpu": "A100 x1", "server_command": "serve model-a", "model_state": "model-a@revision-1"}
    failed = fail_tool.invoke({**identity, "output_details": True, "cache_report": True, "num_prompts": 1, "timeout_s": 3.0})

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
    succeeded = ok_tool.invoke({**identity, "num_prompts": 20})

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

  def test_final_attempt_can_log_before_wrap_up(self) -> None:
    """Finish and log the sole allowed run before the worker summarizes."""
    registry = SessionRegistry()
    registry.add(FakeRemoteSession("agent-10", RemoteResult(exit_status=0, stdout="Request throughput (req/s): 12.0", stderr="")))
    model = ToolBindingFakeModel(responses=[
      AIMessage(content="pod ready"),
      AIMessage(content="setup_ready: commit"),
      AIMessage(content="", tool_calls=[{
        "name": "sglang_bench_serving",
        "args": {"gpu": "H100 x1", "server_command": "serve model-a", "model_state": "model-a@revision-1", "num_prompts": 1},
        "id": "benchmark-call",
        "type": "tool_call",
      }]),
      AIMessage(content="", tool_calls=[{
        "name": "remote_terminal",
        "args": {"command": "echo should-not-run"},
        "id": "remote-call",
        "type": "tool_call",
      }]),
      AIMessage(content="", tool_calls=[{
        "name": "log_experiment",
        "args": {"intent_key": "single capped run", "metrics": {"request_rps": 12.0}},
        "id": "log-call",
        "type": "tool_call",
      }]),
      AIMessage(content="done"),
      AIMessage(content="final summary"),
    ])
    graph = build_subagent_graph(model, 10, registry, self.db_path, "repo", "commit", attempt_cap=1)

    with patch.object(db, "embed_intent", return_value=[1.0, 0.0]):
      result = graph.invoke({"agent_id": 10, "objective": "benchmark goal", "messages": []})

    run = self.run_row(agent_id=10)
    experiments = self.conn.execute("SELECT COUNT(*) FROM experiments WHERE agent_id = 10").fetchone()[0]
    self.assertIsNotNone(run["finished_at"])
    self.assertEqual(experiments, 1)
    self.assertEqual(len(registry.get("agent-10").commands), 1)
    self.assertEqual(result["summary"], "final summary")

  def test_cloud_resource_is_persisted_and_terminated(self) -> None:
    """Keep a provider ID durable until host cleanup succeeds."""
    created = subprocess.CompletedProcess([], 0, stdout='{"id":"pod-123"}\n', stderr="")
    failed_cleanup = subprocess.CompletedProcess([], 1, stdout="", stderr="temporary provider error")
    terminated = subprocess.CompletedProcess([], 0, stdout="deleted pod-123\n", stderr="")
    db.record_agent_session(self.conn, session_name="perferox-agent-11", role="subagent", agent_id=11)
    lambda_tool = provider_cli("lambda", self.db_path, 11)
    with patch("perferox.tools.subprocess.run") as launch:
      count_refused = lambda_tool.invoke({"arguments": ["up", "gpu_1x_a100", "--count=2"]})
    self.assertIn("exactly one Lambda instance", count_refused)
    launch.assert_not_called()

    tool = provider_cli("runpod", self.db_path, 11)
    refused = tool.invoke({"arguments": ["template", "delete", "shared-template"]})
    with patch("perferox.tools.subprocess.run", side_effect=[created, failed_cleanup]) as run:
      output = tool.invoke({"arguments": ["pod", "create", "--image", "image", "--gpu-id", "H100"]})
      cleanup = cleanup_cloud_resource(self.db_path, 11, "secret")

    resource = self.conn.execute("SELECT * FROM agent_sessions WHERE agent_id = 11").fetchone()
    self.assertIn("resource_id=pod-123", output)
    self.assertIn("refused", refused)
    self.assertIn("temporary provider error", cleanup)
    self.assertEqual(resource["resource_id"], "pod-123")
    self.assertEqual(run.call_args_list[1].args[0], ["runpodctl", "pod", "delete", "pod-123"])

    db.finish_agent_session(self.conn, session_name="perferox-agent-11", status="exited")
    with patch("perferox.tools.subprocess.run", return_value=terminated):
      _wait_for_main_event(self.db_path, poll_s=0, cloud_api_key="secret")
    retried = self.conn.execute("SELECT resource_id FROM agent_sessions WHERE agent_id = 11").fetchone()
    self.assertEqual(retried["resource_id"], "")

  def test_modal_sandbox_uses_native_exec_and_host_cleanup(self) -> None:
    """Persist, execute in, and terminate one Modal Sandbox without SSH."""
    with patch.dict(os.environ, {"MODAL_TOKEN_ID": "ak-test", "MODAL_TOKEN_SECRET": "as-test"}, clear=True):
      cloud_key = modal_cloud_key()
    self.assertEqual(cloud_environment(cloud_key), {"MODAL_TOKEN_ID": "ak-test", "MODAL_TOKEN_SECRET": "as-test"})

    process = MagicMock()
    process.stdout.read.return_value = "gpu ready\n"
    process.stderr.read.return_value = ""
    process.wait.return_value = 0
    sandbox = MagicMock(object_id="sb-123")
    sandbox.exec.return_value = process
    recovered_sandbox = MagicMock()
    recovered_sandbox.detach.side_effect = RuntimeError("local detach failed")
    app = object()
    base_image = MagicMock()
    image = object()
    readiness_probe = object()
    base_image.entrypoint.return_value = image
    registry = SessionRegistry()
    db.record_agent_session(self.conn, session_name="perferox-agent-12", role="subagent", agent_id=12)
    tool = create_modal_sandbox(registry, "agent-12", self.db_path, 12)

    with (
      patch("perferox.tools.modal.App.lookup", return_value=app) as lookup_app,
      patch("perferox.tools.modal.Image.from_registry", return_value=base_image) as load_image,
      patch("perferox.tools.modal.Probe.with_exec", return_value=readiness_probe) as make_probe,
      patch("perferox.tools.modal.Sandbox.create", return_value=sandbox) as create_sandbox,
      patch("perferox.tools.modal.Sandbox.from_id", return_value=recovered_sandbox) as load_sandbox,
    ):
      output = tool.invoke({"image": "lmsysorg/sglang:latest", "gpu": "H100"})
      result = registry.get("agent-12").run("bash -lc 'nvidia-smi'", timeout_s=7)
      registry.close("agent-12")
      cleanup = cleanup_cloud_resource(self.db_path, 12)

    resource = self.conn.execute("SELECT provider, resource_id FROM agent_sessions WHERE agent_id = 12").fetchone()
    self.assertIn("resource_id=sb-123", output)
    self.assertEqual(result, RemoteResult(0, "gpu ready\n", ""))
    self.assertEqual(cleanup, "")
    self.assertEqual(dict(resource), {"provider": "modal", "resource_id": ""})
    lookup_app.assert_called_once_with("perferox", create_if_missing=True)
    load_image.assert_called_once_with("lmsysorg/sglang:latest")
    base_image.entrypoint.assert_called_once_with([])
    make_probe.assert_called_once_with("bash", "-lc", "command -v sleep >/dev/null")
    create_sandbox.assert_called_once_with(
      "sleep", "infinity",
      app=app,
      image=image,
      gpu="H100",
      cpu=MODAL_CPU_CORES,
      memory=MODAL_MEMORY_MIB,
      timeout=24 * 60 * 60,
      readiness_probe=readiness_probe,
    )
    sandbox.wait_until_ready.assert_called_once_with(timeout=MODAL_READY_TIMEOUT_S)
    sandbox.exec.assert_called_once_with("bash", "-lc", "nvidia-smi", timeout=7)
    load_sandbox.assert_called_once_with("sb-123")
    sandbox.detach.assert_called_once_with()
    recovered_sandbox.terminate.assert_called_once_with(wait=True)
    recovered_sandbox.detach.assert_called_once_with()

  def test_modal_readiness_failure_terminates_sandbox(self) -> None:
    """Terminate a Sandbox that fails readiness before it becomes host-owned."""
    sandbox = MagicMock(object_id="sb-unready")
    sandbox.wait_until_ready.side_effect = TimeoutError("not ready")
    base_image = MagicMock()
    base_image.entrypoint.return_value = object()
    registry = SessionRegistry()
    db.record_agent_session(self.conn, session_name="perferox-agent-13", role="subagent", agent_id=13)
    tool = create_modal_sandbox(registry, "agent-13", self.db_path, 13)

    with (
      patch("perferox.tools.modal.App.lookup", return_value=object()),
      patch("perferox.tools.modal.Image.from_registry", return_value=base_image),
      patch("perferox.tools.modal.Probe.with_exec", return_value=object()),
      patch("perferox.tools.modal.Sandbox.create", return_value=sandbox),
    ):
      output = tool.invoke({"image": "lmsysorg/sglang:latest", "gpu": "H100"})

    resource_id = self.conn.execute("SELECT resource_id FROM agent_sessions WHERE agent_id = 13").fetchone()[0]
    self.assertIn("Modal Sandbox creation failed: TimeoutError: not ready", output)
    self.assertEqual(resource_id, "")
    with self.assertRaises(KeyError):
      registry.get("agent-13")
    sandbox.terminate.assert_called_once_with(wait=True)
    sandbox.detach.assert_called_once_with()


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

    snapshot = read_dashboard(self.db_path, trace_limit=10)
    delivered = self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"]
    db.take_main_notifications(self.conn)
    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", side_effect=[subprocess.CompletedProcess([], 0, stdout=f"{MAIN_SESSION}\n", stderr=""), subprocess.CompletedProcess([], 1)]):
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


class ProviderRegistryTests(unittest.TestCase):
  """Cover the provider table, the stored profile, and credential resolution."""

  def setUp(self) -> None:
    """Point the profile and credential files at a throwaway directory."""
    self.tempdir = tempfile.TemporaryDirectory()
    root = Path(self.tempdir.name)
    self.addCleanup(self.tempdir.cleanup)
    patches = {
      "CONFIG_DIR": root,
      "CONFIG_PATH": root / "config.json",
      "CREDENTIALS_PATH": root / "credentials.json",
    }
    for name, value in patches.items():
      patcher = patch.object(providers, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)
    for name in ("PERFEROX_PROVIDER", "PERFEROX_CHAT_MODEL", "PERFEROX_CLOUD", "DEEPSEEK_API_KEY", "CLOUDFLARE_ACCOUNT_ID"):
      patcher = patch.dict(os.environ, {}, clear=False)
      patcher.start()
      self.addCleanup(patcher.stop)
      os.environ.pop(name, None)

  def test_provider_rows_are_well_formed(self) -> None:
    """Every row must be uniquely named, addressable, and offer a default model."""
    seen: set[str] = set()
    for provider in providers.PROVIDERS:
      self.assertNotIn(provider.name, seen, f"duplicate provider {provider.name}")
      seen.add(provider.name)
      self.assertTrue(provider.models, f"{provider.name} offers no models")
      self.assertEqual(provider.default_model, provider.models[0])
      self.assertIn(provider.auth, (providers.OAUTH, providers.KEY, providers.LOCAL))
      if provider.auth == providers.KEY:
        self.assertTrue(provider.base_url.startswith("https://"), f"{provider.name} is not HTTPS")
        self.assertFalse(provider.base_url.endswith("/"), f"{provider.name} has a trailing slash")
        self.assertTrue(all(character.isupper() or character == "_" for character in provider.key_env), f"{provider.name} key_env is not UPPER_SNAKE")
      if provider.auth == providers.LOCAL:
        self.assertTrue(provider.base_url.startswith("http://127.0.0.1"), f"{provider.name} is not a loopback server")
    self.assertEqual(len(seen), len(providers.PROVIDER_NAMES))

  def test_settings_round_trip_and_environment_overrides(self) -> None:
    """A saved profile survives a reload; env vars win; a switch resets the model."""
    self.assertIsNone(providers.load_settings())
    providers.save_settings(providers.Settings(provider="deepseek", model="deepseek-v4-flash", cloud="modal"))
    self.assertEqual(providers.load_settings(), providers.Settings(provider="deepseek", model="deepseek-v4-flash", cloud="modal"))
    self.assertEqual(providers.CONFIG_PATH.stat().st_mode & 0o777, 0o600)

    with patch.dict(os.environ, {"PERFEROX_CHAT_MODEL": "deepseek-v4-pro"}):
      self.assertEqual(providers.active_settings().model, "deepseek-v4-pro")
    with patch.dict(os.environ, {"PERFEROX_PROVIDER": "groq"}):
      # Switching providers must not carry a model tag the new provider cannot serve.
      self.assertEqual(providers.active_settings(), providers.Settings(provider="groq", model="openai/gpt-oss-120b", cloud="modal"))

  def test_credentials_prefer_the_environment_and_stay_private(self) -> None:
    """Stored keys are owner-only, and an exported key wins over a stored one."""
    deepseek = providers.find("deepseek")
    providers.save_credential("deepseek", "stored-key")
    self.assertEqual(providers.CREDENTIALS_PATH.stat().st_mode & 0o777, 0o600)
    self.assertEqual(providers.api_key(deepseek), "stored-key")
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
      self.assertEqual(providers.api_key(deepseek), "env-key")

  def test_account_scoped_base_url_needs_an_account_id(self) -> None:
    """Cloudflare's endpoint is unusable until an account id is configured."""
    cloudflare = providers.find("cloudflare")
    with self.assertRaises(ValueError):
      providers.base_url(cloudflare)
    providers.save_credential("cloudflare_account", "acct-123")
    self.assertEqual(providers.base_url(cloudflare), "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/v1")

  def test_auth_readiness_matches_the_credential_actually_present(self) -> None:
    """Local needs nothing, a key provider needs its key, and the hint names it."""
    self.assertTrue(perferox_auth.auth_ready(providers.find("llamacpp")))
    groq = providers.find("groq")
    self.assertFalse(perferox_auth.auth_ready(groq))
    self.assertIn("GROQ_API_KEY", perferox_auth.missing_credential(groq))
    providers.save_credential("groq", "gsk-test")
    self.assertTrue(perferox_auth.auth_ready(groq))
    self.assertIn("perferox login grok", perferox_auth.missing_credential(providers.find("grok")))


class XaiOAuthTests(unittest.TestCase):
  """Cover the parts of the xAI sign-in that fail silently if they regress."""

  def setUp(self) -> None:
    """Keep the token store inside a throwaway directory."""
    self.tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tempdir.cleanup)
    self.token_path = Path(self.tempdir.name) / "xai_oauth.json"
    patcher = patch.object(xai_oauth, "TOKEN_PATH", self.token_path)
    patcher.start()
    self.addCleanup(patcher.stop)

  @staticmethod
  def _jwt(expiry: int) -> str:
    """Build a token whose payload carries only the given `exp` claim."""
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"

  def test_pkce_challenge_is_the_s256_of_the_verifier(self) -> None:
    """A mismatched challenge would only fail at the token exchange, far from here."""
    verifier, challenge = xai_oauth._pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    self.assertEqual(challenge, expected)
    self.assertLessEqual(len(verifier), 128)
    self.assertNotIn("=", verifier)

  def test_credentials_only_go_to_https_x_ai_endpoints(self) -> None:
    """The token endpoint is cached on disk, so it is re-pinned before every use."""
    xai_oauth._validate_xai_https("https://auth.x.ai/oauth2/token")
    for hostile in ("http://auth.x.ai/oauth2/token", "https://auth.x.ai.evil.test/token", "https://example.com/token"):
      with self.assertRaises(ValueError, msg=hostile):
        xai_oauth._validate_xai_https(hostile)

  def test_expiry_leeway_refreshes_before_the_token_actually_dies(self) -> None:
    """A token inside the leeway is stale; an unreadable one is treated as live."""
    now = 1_000_000
    self.assertTrue(xai_oauth.expires_soon(self._jwt(now + xai_oauth.EXPIRY_LEEWAY_S - 1), now))
    self.assertFalse(xai_oauth.expires_soon(self._jwt(now + xai_oauth.EXPIRY_LEEWAY_S + 60), now))
    self.assertFalse(xai_oauth.expires_soon("not-a-jwt", now))

  def test_a_stale_token_is_refreshed_and_a_dead_grant_is_forgotten(self) -> None:
    """One refresh replaces both tokens; a rejected refresh clears the session."""
    now = time()
    xai_oauth.save_tokens(xai_oauth.StoredTokens(self._jwt(int(now) - 10), "refresh-1", "https://auth.x.ai/oauth2/token"))
    self.assertEqual(self.token_path.stat().st_mode & 0o777, 0o600)

    fresh = self._jwt(int(now) + 3600)
    with patch.object(xai_oauth, "_exchange", return_value={"access_token": fresh, "refresh_token": "refresh-2"}) as exchange:
      self.assertEqual(xai_oauth.XaiTokenProvider().get_token(), fresh)
    self.assertEqual(exchange.call_args.args[1]["grant_type"], "refresh_token")
    self.assertEqual(xai_oauth.load_tokens().refresh_token, "refresh-2")

    with patch.object(xai_oauth, "_exchange", side_effect=RuntimeError("HTTP 400")), self.assertRaises(RuntimeError):
      xai_oauth.XaiTokenProvider()._refresh(xai_oauth.StoredTokens("stale", "refresh-2", "https://auth.x.ai/oauth2/token"))
    self.assertIsNone(xai_oauth.load_tokens())
    self.assertFalse(xai_oauth.auth_ready())

  def test_the_bearer_header_is_stamped_at_send_time(self) -> None:
    """The key baked in when the chat model was built must never reach the API."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
      seen.append(request.headers.get("authorization"))
      return httpx.Response(200, json={})

    auth_flow = xai_oauth.BearerAuth(MagicMock(get_token=MagicMock(return_value="fresh")))
    with httpx.Client(auth=auth_flow, transport=httpx.MockTransport(handler)) as client:
      client.post("https://api.x.ai/v1/chat/completions", headers={"Authorization": "Bearer stale"}, json={})
    self.assertEqual(seen, ["Bearer fresh"])


if __name__ == "__main__":
  unittest.main()
