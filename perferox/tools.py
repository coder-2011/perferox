# ruff: noqa: BLE001

import json
import os
import shlex
import signal
import subprocess
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


@tool("local_terminal", description="Run one shell command on the local host. Use for local files and local setup; directory changes do not persist.")
def local_terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
  """Run one shell command on the local host."""
  return run_local_command(command, timeout_s)


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
      with db.open_db(db_path) as conn:
        run_id = db.start_benchmark_run(conn, agent_id=agent_id, command=command, trace_ref=trace_ref, attempt_cap=attempt_cap)
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
    description="Locally mark a successful benchmark run and save normalized metrics to SQLite; use parsed_metrics from sglang_bench_serving.",
  )
  def log_experiment(intent_key: str, metrics: dict[str, float | int | None] | None = None) -> str:
    """Log normalized metrics for the agent's latest successful run."""
    try:
      with db.open_db(db_path) as conn:
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
