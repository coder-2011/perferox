# ruff: noqa: BLE001

import os
import shlex
import signal
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from perferox import db
from perferox.bench import BenchServingArgs, bench_serving_argv
from perferox.remote import RemoteSession, SessionRegistry

DEFAULT_TIMEOUT_S = 30.0
MAX_OUTPUT_CHARS = 10000
MAX_SEARCH_RESULTS = 50
SKIP_SEARCH_DIRS = {".git", ".ruff_cache", ".venv", "__pycache__"}


def search_files_tool(cwd: str | Path) -> BaseTool:
  """Create a fuzzy repository path search tool."""
  root = Path(cwd).resolve()

  @tool("search_files", description="Fuzzy-search repository file paths by name or path, not file contents.")
  def search_files(query: str, path: str = ".", limit: int = MAX_SEARCH_RESULTS) -> str:
    """Return the best matching repository file paths."""
    query_key = "".join(ch for ch in query.casefold() if ch.isalnum())
    if not query_key:
      return "query must contain at least one letter or digit"
    if limit < 1 or limit > MAX_SEARCH_RESULTS:
      return f"limit must be between 1 and {MAX_SEARCH_RESULTS}"
    try:
      search_root = (root / path).resolve()
      search_root.relative_to(root)
    except ValueError:
      return f"path escapes repository root: {path}"
    choices = {}
    paths = [(search_root.parent, [], [search_root.name])] if search_root.is_file() else os.walk(search_root)
    for dirpath, dirnames, filenames in paths:
      dirnames[:] = [name for name in dirnames if name not in SKIP_SEARCH_DIRS]
      for filename in filenames:
        rel_path = (Path(dirpath) / filename).relative_to(root).as_posix()
        path_key = "".join(ch for ch in rel_path.casefold() if ch.isalnum())
        if query_key in path_key:
          choices[rel_path] = 1000 - path_key.index(query_key)
          continue
        last = -1
        score = 0
        for char in query_key:
          index = path_key.find(char, last + 1)
          if index < 0:
            score = 0
            break
          score += 15 if index == last + 1 else 10
          last = index
        if score:
          choices[rel_path] = score
    matches = sorted(choices.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
    return "\n".join(f"score={score} {rel_path}" for rel_path, score in matches[:limit]) or "no matches"

  return search_files


@tool(
  "local_terminal",
  description="Run one shell command on the local host. Use for local files and local setup; directory changes do not persist.",
)
def local_terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
  return _run_local(command, timeout_s)


def connect_remote_session(registry: SessionRegistry, session_id: str) -> BaseTool:
  @tool(
    "connect_remote_session",
    description="Open the persistent SSH session after local runpodctl returns host, user, and port.",
  )
  def connect(host: str, user: str = "root", port: int = 22, timeout_s: float = 30.0) -> str:
    registry.close(session_id)
    try:
      registry.add(RemoteSession.connect(session_id, host, user, port, timeout_s))
    except Exception as exc:
      return f"remote session connect failed: {type(exc).__name__}: {exc}"
    return f"connected remote session {session_id} to {user}@{host}:{port}"

  return connect


def remote_terminal(registry: SessionRegistry, session_id: str) -> BaseTool:
  @tool(
    "remote_terminal",
    description="Run one shell command on the connected remote SSH machine.",
  )
  def terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
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
  @tool(
    "sglang_bench_serving",
    args_schema=BenchServingArgs,
    description="Run one structured SGLang serving benchmark on the connected remote SSH machine and return its host-assigned run_id.",
  )
  def run(**kwargs: Any) -> str:
    """Run one benchmark command after the host assigns its run id."""
    try:
      args = BenchServingArgs(**kwargs)
      command = " ".join(shlex.quote(part) for part in bench_serving_argv(args))
    except Exception as exc:
      return f"invalid bench_serving args: {type(exc).__name__}: {exc}"
    session = registry.get(session_id)
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.start_benchmark_run(conn, agent_id=agent_id, command=command, trace_ref=trace_ref, attempt_cap=attempt_cap)
    except Exception as exc:
      return f"benchmark not started: {type(exc).__name__}: {exc}"
    output = _run_remote(session, command, args.timeout_s)
    if not output.startswith("exit_code=0\n"):
      # Failed started runs still count, so mark them in SQLite.
      with closing(db.connect(db_path)) as conn:
        db.mark_run_failed(conn, agent_id=agent_id, run_id=run_id, error=output)
    return f"run_id={run_id}\n{output}"

  return run


def log_experiment_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  @tool(
    "log_experiment",
    description="Locally mark a successful benchmark run and save its metrics to SQLite.",
  )
  def log_experiment(intent_key: str, metrics: dict[str, float | int | None] | None = None) -> str:
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.log_experiment(conn, agent_id=agent_id, intent_key=intent_key, metrics=metrics)
    except Exception as exc:
      return f"log_experiment failed: {type(exc).__name__}: {exc}"
    return f"logged experiment agent_id={agent_id} run_id={run_id}"

  return log_experiment


def log_anomaly_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  @tool(
    "log_anomaly",
    description="Locally save a human-readable anomaly for a benchmark run to SQLite.",
  )
  def log_anomaly(run_id: int, summary: str) -> str:
    try:
      with closing(db.connect(db_path)) as conn:
        anomaly_id = db.log_anomaly(conn, agent_id=agent_id, run_id=run_id, summary=summary)
    except Exception as exc:
      return f"log_anomaly failed: {type(exc).__name__}: {exc}"
    return f"logged anomaly anomaly_id={anomaly_id} agent_id={agent_id} run_id={run_id}"

  return log_anomaly


def _run_local(command: str, timeout_s: float | None) -> str:
  """Run a local command, killing the process group on timeout."""
  try:
    process = subprocess.Popen(
      ["bash", "-lc", command],
      text=True,
      encoding="utf-8",
      errors="replace",
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      start_new_session=os.name == "posix",
    )
    stdout, stderr = process.communicate(timeout=timeout_s)
    return _format_result(process.returncode, stdout, stderr)
  except subprocess.TimeoutExpired:
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
  output = "\n".join(part for part in (stdout, stderr) if part)
  if len(output) > MAX_OUTPUT_CHARS:
    keep = MAX_OUTPUT_CHARS // 2
    skipped = len(output) - MAX_OUTPUT_CHARS
    output = f"{output[:keep]}\n\n... {skipped} chars elided ...\n\n{output[-keep:]}"
  return f"exit_code={exit_status}\n{output}"
