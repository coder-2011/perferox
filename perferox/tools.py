# ruff: noqa: BLE001

import json
import os
import shlex
import signal
import subprocess
from collections.abc import Mapping
from contextlib import closing
from heapq import nsmallest
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from perferox import db
from perferox.bench import BenchServingArgs, bench_serving_argv, parse_bench_serving_metrics
from perferox.remote import RemoteSession, SessionRegistry

DEFAULT_TIMEOUT_S = 30.0
MAX_OUTPUT_CHARS = 10000
MAX_SEARCH_RESULTS = 50
SKIP_SEARCH_DIRS = {".git", ".ruff_cache", ".venv", "__pycache__"}
WEB_SEARCH_TOOL = {"type": "web_search", "external_web_access": True, "search_context_size": "high"}
_PROVIDER_COMMANDS = {
  "runpod": (
    "runpodctl",
    ("pod", "create"),
    (("--help",), ("doctor",), ("version",), ("gpu", "list"), ("pod", "list"), ("pod", "get"), ("template", "list"), ("template", "search"), ("template", "get"), ("ssh", "info"), ("ssh", "list-keys")),
  ),
  "lambda": ("lambda-labs", ("up",), (("--help",), ("catalog",), ("keys",), ("ls",))),
}


def search_files_tool(cwd: str | Path) -> BaseTool:
  """Create a fuzzy repository path search tool."""
  root = Path(cwd).resolve()
  root_path = os.fspath(root)

  @tool("search_files", description="Fuzzy-search repository file paths by name or path, not file contents.")
  def search_files(query: str, path: str = ".", limit: int = MAX_SEARCH_RESULTS) -> str:
    """Return the best matching repository file paths."""
    query_key = "".join(filter(str.isalnum, query.casefold()))
    if not query_key:
      return "query must contain at least one letter or digit"
    if limit < 1 or limit > MAX_SEARCH_RESULTS:
      return f"limit must be between 1 and {MAX_SEARCH_RESULTS}"
    try:
      search_root = (root / path).resolve()
      search_root.relative_to(root)
    except ValueError:
      return f"path escapes repository root: {path}"
    matches = []
    paths = [(os.fspath(search_root.parent), [], [search_root.name])] if search_root.is_file() else os.walk(os.fspath(search_root))
    for dirpath, dirnames, filenames in paths:
      dirnames[:] = [name for name in dirnames if name not in SKIP_SEARCH_DIRS]
      for filename in filenames:
        rel_path = os.path.relpath(os.path.join(dirpath, filename), root_path)
        if os.sep != "/":
          rel_path = rel_path.replace(os.sep, "/")
        path_key = "".join(filter(str.isalnum, rel_path.casefold()))
        index = path_key.find(query_key)
        if index >= 0:
          matches.append((rel_path, 1000 - index))
          continue
        if len(query_key) > len(path_key):
          continue
        last = -1
        score = 0
        for char in query_key:
          index = path_key.find(char, last + 1)
          if index < 0:
            break
          score += 15 if index == last + 1 else 10
          last = index
        else:
          matches.append((rel_path, score))
    matches = nsmallest(limit, matches, key=lambda item: (-item[1], len(item[0]), item[0]))
    return "\n".join(f"score={score} {rel_path}" for rel_path, score in matches) or "no matches"

  return search_files


def connect_remote_session(registry: SessionRegistry, session_id: str) -> BaseTool:
  """Create the tool that owns one host-assigned SSH session id."""
  @tool("connect_remote_session", description="Open the persistent SSH session after the selected cloud provider returns host, user, and port.")
  def connect(host: str, user: str = "root", port: int = 22, timeout_s: float = 30.0) -> str:
    """Replace the id's SSH connection with a newly connected session."""
    registry.close(session_id)
    try:
      registry.add(RemoteSession.connect(session_id, host, user, port, timeout_s))
    except Exception as exc:
      return f"remote session connect failed: {type(exc).__name__}: {exc}"
    return f"connected remote session {session_id} to {user}@{host}:{port}"

  return connect


def remote_terminal(registry: SessionRegistry, session_id: str) -> BaseTool:
  """Create a shell tool bound to one host-assigned SSH session id."""
  @tool("remote_terminal", description="Run one shell command on the connected remote SSH machine.")
  def terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
    """Run one shell command through the bound SSH session."""
    try:
      session = registry.get(session_id)
    except KeyError:
      return _format_result(None, "", f"remote session not connected: {session_id}")
    return _run_remote(session, command, timeout_s)

  return terminal


def sglang_bench_serving(
  registry: SessionRegistry,
  session_id: str,
  db_path: str | Path,
  agent_id: int,
  repository: str = "",
  commit: str = "",
  provider: str = "",
  trace_ref: str = "",
  attempt_cap: int | None = None,
) -> BaseTool:
  """Create the structured SGLang benchmark tool for one subagent."""
  @tool(
    "sglang_bench_serving", args_schema=BenchServingArgs,
    description="Run one structured SGLang serving benchmark on the connected remote SSH machine and return its host-assigned run_id.",
  )
  def run(**kwargs: Any) -> str:
    """Run one benchmark command after the host assigns its run id."""
    try:
      args = BenchServingArgs(**kwargs)
      command = shlex.join(bench_serving_argv(args))
    except Exception as exc:
      return f"invalid bench_serving args: {type(exc).__name__}: {exc}"
    session = registry.get(session_id)
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.start_benchmark_run(
          conn,
          agent_id=agent_id,
          command=command,
          repository=repository,
          commit=commit,
          provider=provider,
          gpu=args.gpu,
          server_command=args.server_command,
          model_state=args.model_state,
          trace_ref=trace_ref,
          attempt_cap=attempt_cap,
        )
    except Exception as exc:
      return f"benchmark not started: {type(exc).__name__}: {exc}"
    output = _run_remote(session, command, args.timeout_s)
    if not output.startswith("exit_code=0\n"):
      # Failed started runs still count, so mark them in SQLite.
      with closing(db.connect(db_path)) as conn:
        db.mark_run_failed(conn, agent_id=agent_id, run_id=run_id, error=output)
      return f"run_id={run_id}\n{output}"
    metrics = parse_bench_serving_metrics(output, expected_requests=args.num_prompts)
    metrics_json = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    return f"run_id={run_id}\nparsed_metrics={metrics_json}\n{output}"

  return run


def provider_cli(provider: str, db_path: str | Path, agent_id: int) -> BaseTool:
  """Create one bounded provider CLI that records every paid resource."""
  executable, create_prefix, read_prefixes = _PROVIDER_COMMANDS[provider]

  @tool("provider_cli", description=f"Run allowlisted {provider} CLI arguments without a shell; the host records creation and owns teardown.")
  def run(arguments: list[str]) -> str:
    """Run one read or create command through the selected provider CLI."""
    if not arguments or any(not argument for argument in arguments):
      return "provider_cli arguments must be non-empty strings"
    prefix = tuple(arguments)
    creating = prefix[:len(create_prefix)] == create_prefix and "--help" not in arguments
    readable = "--help" in arguments or any(prefix[:len(allowed)] == allowed for allowed in read_prefixes)
    if not creating and not readable:
      return f"provider_cli refused unsupported or mutating {provider} command"
    if provider == "lambda" and creating and "--count" in arguments:
      return "provider_cli permits exactly one Lambda instance"
    if creating:
      with closing(db.connect(db_path)) as conn:
        if db.stop_requested(conn, agent_id=agent_id):
          return "stop requested; provider creation refused"
        if db.pending_cloud_resources(conn, agent_id=agent_id):
          return "provider_cli permits one active cloud resource per subagent"
    argv = [executable, *arguments]
    if provider == "runpod" and creating and not {"--output", "-o"} & set(arguments):
      argv.extend(("--output", "json"))
    result = _run_argv(argv)
    if not creating or not result.startswith("exit_code=0\n"):
      return result
    resource_id = _created_resource_id(provider, result.partition("\n")[2])
    if not resource_id:
      return f"{result}\nprovider_cli could not identify the created resource; inspect provider inventory immediately"
    try:
      with closing(db.connect(db_path)) as conn:
        db.record_cloud_resource(conn, agent_id=agent_id, provider=provider, resource_id=resource_id)
    except Exception as exc:
      cleanup = _terminate_resource(provider, resource_id, os.environ)
      return f"provider resource registration failed: {type(exc).__name__}: {exc}\ncompensating cleanup: {cleanup}"
    return f"resource_id={resource_id}\n{result}"

  return run


def cleanup_cloud_resources(db_path: str | Path, agent_id: int, api_key: str | None = None) -> list[str]:
  """Terminate one worker's persisted resources and return cleanup failures."""
  with closing(db.connect(db_path, readonly=True)) as conn:
    resources = db.pending_cloud_resources(conn, agent_id=agent_id)
  errors = []
  for resource in resources:
    provider = resource["provider"]
    resource_id = resource["resource_id"]
    env = os.environ.copy()
    if api_key:
      env["LAMBDA_API_KEY" if provider == "lambda" else "RUNPOD_API_KEY"] = api_key
    result = _terminate_resource(provider, resource_id, env)
    error = "" if result.startswith("exit_code=0\n") else result
    with closing(db.connect(db_path)) as conn:
      db.finish_cloud_resource(conn, provider=provider, resource_id=resource_id, error=error)
    if error:
      errors.append(f"{provider} {resource_id}: {error}")
  return errors


def _run_argv(argv: list[str], env: Mapping[str, str] | None = None) -> str:
  """Run one argv command without shell interpretation."""
  try:
    result = subprocess.run(argv, text=True, capture_output=True, check=False, env=env)
  except OSError as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")
  return _format_result(result.returncode, result.stdout, result.stderr)


def _created_resource_id(provider: str, output: str) -> str | None:
  """Extract one resource ID from a successful provider response."""
  if provider == "lambda":
    parts = output.split()
    return parts[1] if len(parts) == 2 and parts[0] == "launched" else None
  try:
    resource_id = json.loads(output)["id"]
  except (KeyError, TypeError, ValueError):
    return None
  return str(resource_id)


def _terminate_resource(provider: str, resource_id: str, env: Mapping[str, str]) -> str:
  """Permanently delete one recorded provider resource."""
  argv = ["lambda-labs", "rm", resource_id] if provider == "lambda" else ["runpodctl", "pod", "delete", resource_id]
  return _run_argv(argv, env)


def log_experiment_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  """Create the host-owned successful experiment logging tool."""
  @tool(
    "log_experiment",
    description="Locally mark a successful benchmark run and save normalized metrics to SQLite; use parsed_metrics from sglang_bench_serving.",
  )
  def log_experiment(intent_key: str, metrics: dict[str, float | int | None] | None = None) -> str:
    """Log normalized metrics for the agent's latest successful run."""
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.log_experiment(conn, agent_id=agent_id, intent_key=intent_key, metrics=metrics)
    except Exception as exc:
      return f"log_experiment failed: {type(exc).__name__}: {exc}"
    return f"logged experiment agent_id={agent_id} run_id={run_id}"

  return log_experiment


def log_anomaly_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  """Create the host-owned anomaly logging tool."""
  @tool(
    "log_anomaly",
    description="Locally save a human-readable anomaly for a benchmark run to SQLite.",
  )
  def log_anomaly(run_id: int, summary: str) -> str:
    """Log one human-readable anomaly against an agent run."""
    try:
      with closing(db.connect(db_path)) as conn:
        anomaly_id = db.log_anomaly(conn, agent_id=agent_id, run_id=run_id, summary=summary)
    except Exception as exc:
      return f"log_anomaly failed: {type(exc).__name__}: {exc}"
    return f"logged anomaly anomaly_id={anomaly_id} agent_id={agent_id} run_id={run_id}"

  return log_anomaly


def run_local_command(command: str, timeout_s: float | None, cwd: str | Path | None = None) -> str:
  """Run a local command, killing the process group on timeout."""
  try:
    process = subprocess.Popen(
      ["bash", "-lc", command],
      text=True,
      encoding="utf-8",
      errors="replace",
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      cwd=cwd,
      start_new_session=os.name == "posix",
    )
    stdout, stderr = process.communicate(timeout=timeout_s)
    return _format_result(process.returncode, stdout, stderr)
  except subprocess.TimeoutExpired:
    # Kill the process group so timed-out commands cannot leave child processes running.
    os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
    stdout, stderr = process.communicate()
    return _format_result(None, stdout, f"{stderr}\ntimed out after {timeout_s}s")
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")


def _run_remote(session: RemoteSession, command: str, timeout_s: float | None) -> str:
  """Run a command through SSH."""
  try:
    result = session.run(f"bash -lc {shlex.quote(command)}", timeout_s=timeout_s)
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")
  return _format_result(result.exit_status, result.stdout, result.stderr)


def _format_result(exit_status: int | None, stdout: str, stderr: str) -> str:
  """Format command output for a tool message."""
  output = f"{stdout}\n{stderr}" if stdout and stderr else stdout or stderr
  if len(output) > MAX_OUTPUT_CHARS:
    keep = MAX_OUTPUT_CHARS // 2
    skipped = len(output) - MAX_OUTPUT_CHARS
    output = f"{output[:keep]}\n\n... {skipped} chars elided ...\n\n{output[-keep:]}"
  return f"exit_code={exit_status}\n{output}"
