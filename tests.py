"""High-signal unit tests for Perferox's host-owned contracts."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from lambda_labs import main as lambda_main
from lambda_labs import request as lambda_request
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from modal.stream_type import StreamType
from pydantic import ValidationError

from perferox import db
from perferox.auth import (
  ModelProfile,
  active_model_profile,
  build_chat_model,
  cloud_environment,
  login_model,
  modal_cloud_key,
)
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
  search_files_tool,
  sglang_bench_serving,
)
from perferox.tui import launch_main, request_end, shutdown_perferox


@dataclass(slots=True)
class FakeRemoteSession:
  """Return one fixed remote command result without opening SSH."""

  session_id: str
  result: RemoteResult
  commands: list[str] = field(default_factory=list)

  def run(self, command: str, *, timeout_s: float = 30.0) -> RemoteResult:
    """Return the configured remote result."""
    self.commands.append(command)
    return self.result


class ToolBindingFakeModel(FakeMessagesListChatModel):
  """Let deterministic test messages pass through LangChain tool binding."""

  def bind_tools(self, tools: Any, **kwargs: Any) -> ToolBindingFakeModel:
    """Return this fake because its responses already contain tool calls."""
    return self


class ModelProfileTests(unittest.TestCase):
  """Protect the CLI-validated model selection."""

  def test_login_replaces_profile_only_after_validation(self) -> None:
    """Accept arbitrary models while preserving the active validated profile."""
    with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"XDG_CONFIG_HOME": root}):
      config_dir = Path(root) / "perferox"
      config_dir.mkdir()
      (config_dir / "provider").write_text("chatgpt", encoding="utf-8")
      self.assertEqual(active_model_profile().model, "chatgpt/gpt-5.4")

      probe = AIMessage(content="", tool_calls=[{"name": "perferox_auth_probe", "args": {}, "id": "probe", "type": "tool_call"}])
      chat_model = ToolBindingFakeModel(responses=[probe])
      with patch("langchain_litellm.ChatLiteLLM", return_value=chat_model):
        logged_in = login_model("anthropic/claude-sonnet", "high")
      self.assertEqual(active_model_profile(), logged_in)

      failed_model = ToolBindingFakeModel(responses=[AIMessage(content="no tool call")])
      with patch("langchain_litellm.ChatLiteLLM", return_value=failed_model), self.assertRaisesRegex(RuntimeError, "did not complete the tool-call probe"):
        login_model("openrouter/auto")
      self.assertEqual(active_model_profile(), logged_in)

  def test_chatgpt_payload_drops_only_unencrypted_reasoning(self) -> None:
    """Keep tool calls and encrypted reasoning while removing invalid Codex stubs."""
    message = AIMessage(content=[
      {"type": "reasoning", "id": "rs_empty", "encrypted_content": ""},
      {"type": "reasoning", "id": "rs_replayable", "encrypted_content": "ciphertext"},
      {"type": "function_call", "id": "fc_test"},
    ])
    model = build_chat_model(ModelProfile("chatgpt/test"))

    with patch.object(model, "_codex_headers_sync", return_value={}):
      payload = model._get_request_payload([message])

    self.assertEqual([item["id"] for item in payload["input"]], ["rs_replayable", "fc_test"])
    self.assertEqual(len(message.content), 3)


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
      python_executable="/workspace/venv/bin/python",
      num_prompts=8,
      request_rate=2.5,
      extra_request_body={"mode": "stress", "seed": 7},
      header={"x-trace": "perferox", "accept": "json"},
      timeout_s=12.0,
    )
    argv = bench_serving_argv(args)

    self.assertEqual(argv[:3], ["/workspace/venv/bin/python", "-m", "sglang.benchmark.serving"])
    self.assertIn("--output-details", argv)
    self.assertIn("--cache-report", argv)
    self.assertEqual(argv[argv.index("--num-prompts") + 1], "8")
    self.assertEqual(argv[argv.index("--extra-request-body") + 1], '{"mode":"stress","seed":7}')
    header_index = argv.index("--header")
    self.assertEqual(argv[header_index + 1:header_index + 3], ["accept=json", "x-trace=perferox"])
    self.assertNotIn("--timeout-s", argv)
    self.assertNotIn("--server-command", argv)
    self.assertNotIn("--python-executable", argv)
    equivalent = args.model_copy(update={"extra_request_body": {"seed": 7, "mode": "stress"}, "header": {"accept": "json", "x-trace": "perferox"}})
    self.assertEqual(bench_serving_argv(equivalent), argv)

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

  def test_termination_requires_provider_confirmation(self) -> None:
    """Do not report success when Lambda leaves the requested instance live."""
    with patch.dict(os.environ, {"LAMBDA_API_KEY": "test-key"}), patch("lambda_labs.request", return_value={"terminated_instances": []}):
      self.assertEqual(lambda_main(["rm", "instance-1"]), 1)


class HostStateTests(DatabaseTestCase):
  """Protect deterministic SQLite-owned state transitions."""

  def test_schema_migration_is_serialized(self) -> None:
    """Keep concurrent legacy upgrades from adding the same column twice."""
    legacy_path = Path(self.tempdir.name) / "legacy.sqlite"
    with closing(sqlite3.connect(legacy_path)) as conn:
      conn.executescript(
        "CREATE TABLE runs(agent_id INTEGER, run_id INTEGER, started_at TEXT, PRIMARY KEY(agent_id, run_id));"
        "CREATE TABLE agent_sessions(session_name TEXT PRIMARY KEY, status TEXT);"
      )
    first_paused = threading.Event()
    second_alter = threading.Event()
    release_first = threading.Event()

    def migrate(worker: int) -> None:
      """Pause the first legacy ALTER while another connection attempts migration."""
      with closing(db.connect(legacy_path)) as conn:
        def trace(sql: str) -> None:
          """Expose whether both connections enter the same legacy ALTER."""
          if not sql.startswith("ALTER TABLE runs ADD COLUMN repository"):
            return
          if worker == 0:
            first_paused.set()
            release_first.wait(2)
          else:
            second_alter.set()

        conn.set_trace_callback(trace)
        db.init_db(conn)

    with ThreadPoolExecutor(max_workers=2) as pool:
      first = pool.submit(migrate, 0)
      self.assertTrue(first_paused.wait(2))
      second = pool.submit(migrate, 1)
      self.assertFalse(second_alter.wait(0.2))
      release_first.set()
      first.result()
      second.result()

    with closing(db.connect(legacy_path)) as conn:
      self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
      self.assertIn("repository", {row["name"] for row in conn.execute("PRAGMA table_info(runs)")})

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

  def test_search_files_reuses_one_scoped_repository_snapshot(self) -> None:
    """Keep scoped results stable after the read-only source index is built."""
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "src"
      source.mkdir()
      (source / "attention_kernel.py").touch()
      search = search_files_tool(root)
      (source / "late_attention.py").touch()

      directory_result = search.invoke({"query": "attention", "path": "src"})
      file_result = search.invoke({"query": "kernel", "path": "src/attention_kernel.py"})
      self.assertEqual(directory_result, "score=997 src/attention_kernel.py")
      self.assertEqual(file_result, "score=988 src/attention_kernel.py")

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
    self.assertIn("serve model-a", registry.get("fail").commands[0])
    self.assertIn('parsed_metrics={"cache_hit_rate":0.75,"error_rate":0.1,"request_rps":12.34}', succeeded)

  def test_experiment_logging_similarity_and_anomalies(self) -> None:
    """Batch pending intents after restart and search their packed vectors."""
    with self.assertRaisesRegex(ValueError, "unknown, unsuccessful, or annotated run"):
      db.log_experiment(self.conn, agent_id=3, run_id=0, intent_key="no run")

    first_run = db.start_benchmark_run(self.conn, agent_id=3, command="valid benchmark")
    db.mark_run_succeeded(self.conn, agent_id=3, run_id=first_run, metrics={"cache_hit_rate": 0.75, "error_rate": 0.25})
    second_run = db.start_benchmark_run(self.conn, agent_id=3, command="scheduler benchmark")
    db.mark_run_succeeded(self.conn, agent_id=3, run_id=second_run, metrics={})

    db.log_experiment(self.conn, agent_id=3, run_id=first_run, intent_key="CUDA cache throughput")
    db.log_experiment(self.conn, agent_id=3, run_id=second_run, intent_key="scheduler tail latency")
    pending = self.conn.execute("SELECT intent_embedding FROM experiments WHERE agent_id = 3 ORDER BY run_id").fetchall()
    self.assertEqual([row["intent_embedding"] for row in pending], [b"", b""])

    with patch.object(db, "_embed_intents", return_value=([1.0, 0.0], [0.0, 1.0])) as embed, closing(db.connect(self.db_path)) as coordinator_conn:
      self.assertEqual(db.embed_pending_intents(coordinator_conn), 2)
      self.assertEqual(db.embed_pending_intents(coordinator_conn), 0)
    embed.assert_called_once_with(("CUDA cache throughput", "scheduler tail latency"))
    with patch.object(db, "embed_intent", return_value=[0.9, 0.1]):
      matches = db.find_similar_experiments(self.conn, "cache-ish intent", limit=2)

    anomaly_id = db.log_anomaly(self.conn, agent_id=3, run_id=0, summary="cache-hit collapse on MI250")
    experiment = self.conn.execute("SELECT * FROM experiments WHERE agent_id = 3 AND run_id = 0").fetchone()
    anomaly = self.conn.execute("SELECT * FROM anomalies WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    self.assertEqual(experiment["intent_embedding"], db._pack_embedding([1.0, 0.0]))
    self.assertEqual(self.conn.execute("SELECT typeof(intent_embedding) FROM experiments WHERE agent_id = 3 AND run_id = 0").fetchone()[0], "blob")
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
      AIMessage(content="", tool_calls=[{
        "name": "remote_terminal",
        "args": {"command": "python -m sglang.benchmark.serving --num-prompts 999"},
        "id": "raw-benchmark-call",
        "type": "tool_call",
      }]),
      AIMessage(content="setup_ready: commit"),
      AIMessage(content="", tool_calls=[{
        "name": "sglang_bench_serving",
        "args": {"gpu": "H100 x1", "server_command": "serve model-a", "model_state": "model-a@revision-1", "num_prompts": 1},
        "id": "benchmark-call",
        "type": "tool_call",
      }]),
      AIMessage(content="", tool_calls=[{
        "name": "log_experiment",
        "args": {"run_id": 0, "intent_key": "single capped run"},
        "id": "log-call",
        "type": "tool_call",
      }]),
      AIMessage(content="done"),
      AIMessage(content="final summary"),
    ])
    graph = build_subagent_graph(model, 10, registry, self.db_path, "repo", "commit", attempt_cap=1)

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
    null_id = subprocess.CompletedProcess([], 0, stdout='{"id":null}\n', stderr="")
    with patch("perferox.tools.subprocess.run", return_value=null_id):
      unidentified = tool.invoke({"arguments": ["pod", "create", "--image", "image", "--gpu-id", "H100"]})
    with patch("perferox.tools.subprocess.run", side_effect=[created, failed_cleanup]) as run:
      output = tool.invoke({"arguments": ["pod", "create", "--image", "image", "--gpu-id", "H100"]})
      cleanup = cleanup_cloud_resource(self.db_path, 11, "secret")

    resource = self.conn.execute("SELECT * FROM agent_sessions WHERE agent_id = 11").fetchone()
    self.assertIn("resource_id=pod-123", output)
    self.assertIn("refused", refused)
    self.assertIn("could not identify", unidentified)
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
    process.stdout = ["gpu ready\n"]
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
    sandbox.exec.assert_called_once_with("bash", "-lc", "nvidia-smi", timeout=7, stderr=StreamType.STDOUT)
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

  def test_modal_shutdown_stops_agents_and_sandboxes(self) -> None:
    """Run both host-owned shutdown actions when Perferox closes."""
    with patch("perferox.tui.request_end") as stop, patch("perferox.tools.cleanup_modal_sandboxes") as cleanup:
      shutdown_perferox(self.db_path)
    stop.assert_called_once_with(self.db_path)
    cleanup.assert_called_once_with(self.db_path)


class TUIWiringTests(DatabaseTestCase):
  """Protect the TUI bridge without model, browser, SSH, or cloud work."""

  def test_dashboard_trace_tail_and_soft_stop_flow(self) -> None:
    """Read live state, preserve notifications, then request soft stop."""
    trace_path = Path(self.tempdir.name) / "main.jsonl"
    lines = [json.dumps({"payload": {"main": {"messages": [{"content": f"cache pressure {index}"}]}}}, separators=(",", ":")) for index in range(30)]
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db.record_agent_session(self.conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path), llm_model="anthropic/claude-sonnet", reasoning_effort="high")
    db.record_agent_session(self.conn, session_name="perferox-agent-0", role="subagent", agent_id=0, trace_ref=str(trace_path))
    db.append_explorer_state(self.conn, agent_id=None, line="explorer saw cache pressure")
    db.start_benchmark_run(self.conn, agent_id=0, command="bench cache")
    db.log_anomaly(self.conn, agent_id=0, run_id=0, summary="cache pressure anomaly")

    with patch("perferox.status.shutil.which", return_value=None):
      snapshot = read_dashboard(self.db_path, trace_limit=10)
    delivered = self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"]
    alive = subprocess.CompletedProcess([], 0, stdout=f"{MAIN_SESSION}\nperferox-agent-0\n", stderr="")
    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", return_value=alive):
      notification_update = _wait_for_main_event(self.db_path, poll_s=0)
    message, notification_ids = notification_update
    self.assertIn("run_started", message)
    self.assertIsNone(self.conn.execute("SELECT delivered_at FROM main_notifications ORDER BY notification_id LIMIT 1").fetchone()["delivered_at"])
    db.acknowledge_main_notifications(self.conn, notification_ids)
    with patch("perferox.status.shutil.which", return_value="tmux"), patch("perferox.status.subprocess.run", side_effect=[subprocess.CompletedProcess([], 0, stdout=f"{MAIN_SESSION}\n", stderr=""), subprocess.CompletedProcess([], 1)]):
      stopped = request_end(self.db_path)
      update = _wait_for_main_event(self.db_path, poll_s=0)

    launched = subprocess.CompletedProcess([], 0, stdout="started", stderr="")
    with patch("perferox.tui.subprocess.run", return_value=launched) as run:
      launch_main(self.tempdir.name, self.db_path, Path(self.tempdir.name) / "traces", "objective", "secret")

    trace_text = "\n".join(snapshot.trace_lines)
    tail_lines = read_trace_tail([str(trace_path)], 5)
    subagent = next(session for session in snapshot.sessions if session["session_name"] == "perferox-agent-0")
    self.assertEqual(snapshot.main_status, "running")
    self.assertEqual(next(session for session in snapshot.sessions if session["role"] == "main")["llm_model"], "anthropic/claude-sonnet")
    self.assertEqual(snapshot.runs, 1)
    self.assertEqual(snapshot.running_runs, 1)
    self.assertEqual(snapshot.anomaly_count, 1)
    self.assertEqual(snapshot.recent_runs[0]["label"], "bench cache")
    self.assertEqual(subagent["run_count"], 1)
    self.assertEqual(snapshot.anomalies[0]["summary"], "cache pressure anomaly")
    self.assertIn("cache pressure 29", trace_text)
    self.assertIsNone(delivered)
    self.assertIn("cache pressure 25", tail_lines[0])
    self.assertIn("cache pressure 29", tail_lines[-1])
    self.assertEqual(stopped, 2)
    self.assertIsNone(update)
    self.assertIn("worker tmux session disappeared", self.run_row(agent_id=0)["error"])
    self.assertEqual(run.call_args.args[0][0], sys.executable)


if __name__ == "__main__":
  unittest.main()
