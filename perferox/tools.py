# ruff: noqa: BLE001

import json
import os
import re
import shlex
import signal
import subprocess
from heapq import nsmallest
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from perferox import db
from perferox.bench import BenchmarkRunArgs, BenchServingArgs, ExperimentMetrics, bench_serving_argv, parse_bench_serving_metrics
from perferox.remote import RemoteSession, SessionRegistry

DEFAULT_TIMEOUT_S = 30.0
REMOTE_TIMEOUT_S = 30 * 60.0
MAX_REMOTE_TIMEOUT_S = 2 * 60 * 60.0
MAX_PROVIDER_TIMEOUT_S = 5 * 60.0
MAX_RESOURCES_PER_AGENT = 2
MAX_OUTPUT_CHARS = 10000
MAX_SEARCH_RESULTS = 50
SKIP_SEARCH_DIRS = {".git", ".ruff_cache", ".venv", "__pycache__"}
WEB_SEARCH_TOOL = {"type": "web_search", "external_web_access": True, "search_context_size": "high"}
_PROVIDER_READ_PREFIXES = {
  "runpod": (("doctor",), ("gpu", "list"), ("pod", "list"), ("pod", "get"), ("ssh", "info"), ("ssh", "list-keys"), ("template", "list"), ("template", "search"), ("template", "get")),
  "lambda": (("catalog",), ("keys",), ("ls",)),
}
_PROVIDER_CREATE_PREFIX = {"runpod": ("pod", "create"), "lambda": ("up",)}
_PROVIDER_DELETE_PREFIX = {"runpod": ("pod", "delete"), "lambda": ("rm",)}


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
    if timeout_s <= 0 or timeout_s > 120:
      return "timeout_s must be between 0 and 120 seconds"
    registry.close(session_id)
    try:
      registry.add(RemoteSession.connect(session_id, host, user, port, timeout_s))
    except Exception as exc:
      return f"remote session connect failed: {type(exc).__name__}: {exc}"
    return f"connected remote session {session_id} to {user}@{host}:{port}"

  return connect


def remote_terminal(registry: SessionRegistry, session_id: str) -> BaseTool:
  """Create a shell tool bound to one host-assigned SSH session id."""
  @tool("remote_terminal", description="Run one command on the connected machine. Each call is a fresh shell: cd, exports, and virtualenv activation do not persist. Prefix each command with its cwd/environment. Timeouts terminate the remote process group.")
  def terminal(command: str, timeout_s: float = REMOTE_TIMEOUT_S) -> str:
    """Run one shell command through the bound SSH session."""
    if timeout_s <= 0 or timeout_s > MAX_REMOTE_TIMEOUT_S:
      return f"timeout_s must be between 0 and {MAX_REMOTE_TIMEOUT_S:g} seconds"
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
  repository: str,
  target_commit: str,
  trace_ref: str = "",
  attempt_cap: int | None = None,
) -> BaseTool:
  """Create the structured SGLang benchmark tool for one subagent."""
  @tool(
    "sglang_bench_serving", args_schema=BenchmarkRunArgs,
    description="Run one bounded SGLang serving benchmark against an already-ready server. Requires a concise intent, observed hardware, exact server launch config, request count, and endpoint/model settings; returns the host-assigned run_id and parsed metrics.",
  )
  def run(**kwargs: Any) -> str:
    """Run one benchmark command after the host assigns its run id."""
    try:
      request = BenchmarkRunArgs(**kwargs)
      identity = request.model_dump(include={"intent_key", "hardware_config", "server_config"})
      common = request.model_dump(exclude={*identity, "advanced_args"}, exclude_none=True)
      overlap = sorted(set(common) & set(request.advanced_args))
      if overlap:
        raise ValueError(f"advanced_args repeats common fields: {', '.join(overlap)}")
      args = BenchServingArgs(**common, **request.advanced_args)
    except Exception as exc:
      return f"invalid bench_serving args: {type(exc).__name__}: {exc}"
    try:
      session = registry.get(session_id)
      probe = session.run("python -c \"import importlib.util; raise SystemExit(not importlib.util.find_spec('sglang.benchmark.serving'))\"", timeout_s=30)
      module = "sglang.benchmark.serving" if probe.exit_status == 0 else "sglang.bench_serving"
      command = shlex.join(bench_serving_argv(args, module))
      with db.open_db(db_path) as conn:
        resource = db.active_cloud_resources(conn, agent_id=agent_id)
        if len(resource) != 1:
          raise ValueError(f"expected one active cloud resource, found {len(resource)}")
        run_id = db.start_benchmark_run(
          conn,
          agent_id=agent_id,
          repository=repository,
          target_commit=target_commit,
          provider=resource[0]["provider"],
          resource_config=resource[0]["environment"],
          hardware_config=identity["hardware_config"],
          server_config=identity["server_config"],
          command=command,
          intent_key=identity["intent_key"],
          trace_ref=trace_ref,
          attempt_cap=attempt_cap,
        )
    except Exception as exc:
      return f"benchmark not started: {type(exc).__name__}: {exc}"
    output = _run_remote(session, command, args.timeout_s)
    if not output.startswith("exit_code=0\n"):
      # Failed started runs still count, so mark them in SQLite.
      with db.open_db(db_path) as conn:
        db.mark_run_failed(conn, agent_id=agent_id, run_id=run_id, error=output)
      return f"run_id={run_id}\n{output}"
    metrics = parse_bench_serving_metrics(output, expected_requests=args.num_prompts)
    metrics_json = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    return f"run_id={run_id}\nparsed_metrics={metrics_json}\n{output}"

  return run


def log_experiment_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  """Create the host-owned successful experiment logging tool."""
  @tool(
    "log_experiment",
    description="Finalize exactly one successful run_id with the non-empty typed parsed_metrics returned by sglang_bench_serving.",
  )
  def log_experiment(run_id: int, metrics: ExperimentMetrics) -> str:
    """Log normalized metrics for the explicitly selected run."""
    try:
      with db.open_db(db_path) as conn:
        logged_run_id = db.log_experiment(conn, agent_id=agent_id, run_id=run_id, metrics=metrics.model_dump(exclude_none=True))
    except Exception as exc:
      return f"log_experiment failed: {type(exc).__name__}: {exc}"
    return f"logged experiment agent_id={agent_id} run_id={logged_run_id}"

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
      with db.open_db(db_path) as conn:
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


def provider_cli(provider: str, db_path: str | Path, agent_id: int) -> BaseTool:
  """Create a bounded provider CLI tool that records every paid resource."""
  executable = "runpodctl" if provider == "runpod" else "lambda-labs"

  @tool("provider_cli", description=f"Run the selected {provider} CLI without a shell. Pass arguments only, excluding `{executable}`. Reads, bounded creation, and deletion of the recorded resource are allowed; the host records every resource, owns final teardown, and reports cleanup failures.")
  def run(arguments: list[str], timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Execute one allowlisted provider command and persist a created resource ID."""
    if timeout_s <= 0 or timeout_s > MAX_PROVIDER_TIMEOUT_S:
      return f"timeout_s must be between 0 and {MAX_PROVIDER_TIMEOUT_S:g} seconds"
    if not arguments or any(not isinstance(argument, str) or not argument for argument in arguments):
      return "provider_cli arguments must be non-empty strings"
    prefix = tuple(arguments[:2])
    create_prefix = _PROVIDER_CREATE_PREFIX[provider]
    delete_prefix = _PROVIDER_DELETE_PREFIX[provider]
    creating = tuple(arguments[:len(create_prefix)]) == create_prefix and "--help" not in arguments
    deleting = tuple(arguments[:len(delete_prefix)]) == delete_prefix and "--help" not in arguments
    readable = "--help" in arguments or arguments[0] in {"--help", "version"} or any(prefix[:len(allowed)] == allowed for allowed in _PROVIDER_READ_PREFIXES[provider])
    if not creating and not deleting and not readable:
      return f"provider_cli refused unsupported or mutating {provider} command"
    if provider == "lambda" and creating and _lambda_count(arguments) != 1:
      return "provider_cli permits exactly one Lambda instance"
    with db.open_db(db_path) as conn:
      active = db.active_cloud_resources(conn, agent_id=agent_id)
      if creating and db.stop_requested(conn, agent_id=agent_id):
        return "stop requested; provider creation refused"
      if creating and active:
        return "provider_cli permits one active cloud resource per subagent"
      if creating:
        created_count = conn.execute("SELECT COUNT(*) FROM cloud_resources WHERE agent_id = ?", (agent_id,)).fetchone()[0]
        if created_count >= MAX_RESOURCES_PER_AGENT:
          return f"provider_cli resource cap reached ({created_count}/{MAX_RESOURCES_PER_AGENT})"
      if deleting:
        resource_id = arguments[len(delete_prefix)] if len(arguments) > len(delete_prefix) else ""
        if len(arguments) != len(delete_prefix) + 1 or len(active) != 1 or resource_id != active[0]["resource_id"]:
          return "provider_cli may delete only this subagent's recorded active resource"
    argv = [executable, *arguments]
    if provider == "runpod" and creating and "--output" not in arguments and "-o" not in arguments:
      argv.extend(("--output", "json"))
    result = _run_argv(argv, timeout_s)
    if deleting and result.startswith("exit_code=0\n"):
      with db.open_db(db_path) as conn:
        db.finish_cloud_resource(conn, provider=provider, resource_id=resource_id)
      return result
    if not creating or not result.startswith("exit_code=0\n"):
      return result
    resource_id = _created_resource_id(provider, result.partition("\n")[2])
    if not resource_id:
      return f"{result}\nprovider_cli could not identify the created resource; inspect provider inventory immediately"
    environment = {"arguments": arguments}
    try:
      with db.open_db(db_path) as conn:
        db.register_cloud_resource(conn, agent_id=agent_id, provider=provider, resource_id=resource_id, environment=environment)
    except Exception as exc:
      cleanup = _terminate_resource(provider, resource_id, os.environ)
      return f"provider resource registration failed: {type(exc).__name__}: {exc}\ncompensating cleanup: {cleanup}"
    return f"resource_id={resource_id}\n{result}"

  return run


def cleanup_cloud_resources(db_path: str | Path, agent_id: int, api_key: str | None = None) -> list[str]:
  """Terminate every recorded live resource for one worker and persist the outcome."""
  with db.open_db(db_path) as conn:
    resources = db.active_cloud_resources(conn, agent_id=agent_id)
  errors = []
  for resource in resources:
    env = os.environ.copy()
    if api_key:
      env["RUNPOD_API_KEY" if resource["provider"] == "runpod" else "LAMBDA_API_KEY"] = api_key
    result = _terminate_resource(resource["provider"], resource["resource_id"], env)
    error = "" if result.startswith("exit_code=0\n") else result
    with db.open_db(db_path) as conn:
      db.finish_cloud_resource(conn, provider=resource["provider"], resource_id=resource["resource_id"], error=error)
    if error:
      errors.append(f"{resource['provider']} {resource['resource_id']}: {error}")
  return errors


def _run_argv(argv: list[str], timeout_s: float | None, env: dict[str, str] | None = None) -> str:
  """Run an argv command with bounded output and no shell interpretation."""
  try:
    result = subprocess.run(argv, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout_s, check=False, env=env)
    return _format_result(result.returncode, result.stdout, result.stderr)
  except subprocess.TimeoutExpired as exc:
    return _format_result(None, exc.stdout or "", f"{exc.stderr or ''}\ntimed out after {timeout_s}s")
  except OSError as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")


def _created_resource_id(provider: str, output: str) -> str | None:
  """Extract the resource ID from one successful provider creation response."""
  if provider == "lambda":
    match = re.search(r"\blaunched\s+([^,\s]+)", output)
    return match.group(1) if match else None
  try:
    payload = json.loads(output)
  except json.JSONDecodeError:
    return None
  if isinstance(payload, dict):
    if isinstance(payload.get("id"), str):
      return payload["id"]
    pod = payload.get("pod")
    if isinstance(pod, dict) and isinstance(pod.get("id"), str):
      return pod["id"]
  return None


def _lambda_count(arguments: list[str]) -> int:
  """Read Lambda's optional count flag without accepting malformed values."""
  if "--count" not in arguments:
    return 1
  index = arguments.index("--count") + 1
  try:
    return int(arguments[index])
  except (IndexError, ValueError):
    return 0


def _terminate_resource(provider: str, resource_id: str, env: dict[str, str]) -> str:
  """Run the provider's deterministic deletion command for one recorded ID."""
  argv = ["runpodctl", "pod", "delete", resource_id] if provider == "runpod" else ["lambda-labs", "rm", resource_id]
  return _run_argv(argv, DEFAULT_TIMEOUT_S, env)


def _run_remote(session: RemoteSession, command: str, timeout_s: float | None) -> str:
  """Run a command through SSH."""
  try:
    result = session.run(command, timeout_s=timeout_s)
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")
  if result.exit_status in {None, 124} and timeout_s is not None:
    return _format_result(None, result.stdout, f"{result.stderr}\ntimed out after {timeout_s}s")
  return _format_result(result.exit_status, result.stdout, result.stderr)


def _format_result(exit_status: int | None, stdout: str, stderr: str) -> str:
  """Format command output for a tool message."""
  output = f"{stdout}\n{stderr}" if stdout and stderr else stdout or stderr
  if len(output) > MAX_OUTPUT_CHARS:
    keep = MAX_OUTPUT_CHARS // 2
    skipped = len(output) - MAX_OUTPUT_CHARS
    output = f"{output[:keep]}\n\n... {skipped} chars elided ...\n\n{output[-keep:]}"
  return f"exit_code={exit_status}\n{output}"
