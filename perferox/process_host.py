"""Host tmux-wrapped Perferox agent processes."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.text import Text

from perferox import db
from perferox.auth import build_chat_model, cloud_provider, read_cloud_key, write_cloud_key
from perferox.main_agent import build_main_agent_graph
from perferox.prompts import CREATE_POD_SYSTEM_PROMPT, LAMBDA_CREATE_POD_SYSTEM_PROMPT
from perferox.remote import SessionRegistry
from perferox.subagent import build_subagent_graph, stream_with_trace
from perferox.tools import cleanup_cloud_resources, provider_cli, reconcile_tmux_sessions

MAIN_SESSION = "perferox-main"
CONSOLE = Console()
ERROR_CONSOLE = Console(stderr=True)


def main(argv: list[str] | None = None, *, cloud_api_key: str | None = None) -> int:
  """Parse the runner command and run the requested whole-agent process."""
  parser = argparse.ArgumentParser(prog="python -m perferox.process_host")
  subparsers = parser.add_subparsers(dest="command", required=True)
  for name in ("launch-main", "main"):
    subparser = subparsers.add_parser(name)
    subparser.add_argument("--db-path", required=True)
    subparser.add_argument("--trace-dir", default="traces")
    subparser.add_argument("--objective", required=True)
    subparser.add_argument("--cwd", default=".")
  subparsers.choices["main"].add_argument("--cloud-key-file", required=True)
  subparsers.choices["main"].add_argument("--poll-s", type=float, default=5.0)
  subagent = subparsers.add_parser("subagent")
  for name in ("agent-id", "db-path", "trace-path", "goal-file", "repository", "commit", "attempt-cap", "cloud-key-file"):
    subagent.add_argument(f"--{name}", required=True)
  args = parser.parse_args(argv)

  if args.command in ("launch-main", "main"):
    cwd = Path(args.cwd).resolve()
    db_path = (cwd / args.db_path).resolve()
    trace_dir = (cwd / args.trace_dir).resolve()

  if args.command == "launch-main":
    tmux = shutil.which("tmux")
    if tmux is None:
      ERROR_CONSOLE.print("[bold red]error:[/] tmux is not installed or not on PATH")
      return 1
    trace_dir.mkdir(parents=True, exist_ok=True)
    if subprocess.run([tmux, "has-session", "-t", MAIN_SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0:
      CONSOLE.print(f"[yellow]{MAIN_SESSION} already running[/] · attach with [bold]tmux attach -t {MAIN_SESSION}[/]")
      return 1
    api_key = cloud_api_key or sys.stdin.read().strip()
    cloud_provider(api_key)
    key_path = write_cloud_key(api_key)
    with db.open_db(db_path) as conn:
      db.init_db(conn)
      # Register before tmux starts so an immediate End request cannot be lost.
      db.finish_agent_session(conn, session_name=MAIN_SESSION, status="missing")
      db.record_agent_session(conn, session_name=MAIN_SESSION, role="main")
    command = shlex.join([
      sys.executable, "-m", "perferox.process_host", "main",
      "--db-path", str(db_path), "--trace-dir", str(trace_dir),
      "--objective", args.objective, "--cwd", str(cwd),
      "--cloud-key-file", str(key_path),
    ])
    try:
      result = subprocess.run([tmux, "new-session", "-d", "-s", MAIN_SESSION, "-c", str(cwd), "--", "bash", "-lc", command], text=True, capture_output=True, check=False)
    except OSError:
      # Delete a secret handoff that tmux never delivered.
      key_path.unlink(missing_ok=True)
      with db.open_db(db_path) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status="missing")
      raise
    if result.returncode != 0:
      key_path.unlink(missing_ok=True)
      with db.open_db(db_path) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status="missing")
    if result.returncode == 0:
      CONSOLE.print(f"[green]started {MAIN_SESSION}[/] · attach with [bold]tmux attach -t {MAIN_SESSION}[/]")
    else:
      ERROR_CONSOLE.print(Text((result.stderr or result.stdout).strip(), style="red"))
    return result.returncode

  if args.command == "main":
    api_key = read_cloud_key(args.cloud_key_file)
    provider = cloud_provider(api_key)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{MAIN_SESSION}-{time.time_ns()}.jsonl"
    workspace = cwd / "sglang"
    status = "exited"
    try:
      with db.open_db(db_path) as conn:
        db.init_db(conn)
        db.record_agent_session(conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path))
        session = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()
        if session["status"] == "ending":
          return 0
      # Preserve the shared checkout so agents can keep branches, commits, and edits.
      if not (workspace / ".git").is_dir():
        subprocess.run(["git", "clone", "https://github.com/sgl-project/sglang.git", str(workspace)], check=True)
      checkpoint_path = db_path.with_suffix(".checkpoints.sqlite")
      with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        checkpointer.setup()
        graph = build_main_agent_graph(
          build_chat_model(role="main"), db_path,
          cloud_provider=provider, cloud_api_key=api_key,
          cwd=workspace, runtime_cwd=cwd, trace_dir=trace_dir,
          checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": trace_path.stem}}
        initial_state = {"objective": args.objective, "messages": []}
        graph_input = None if checkpointer.get_tuple(config) else initial_state
        pending_notifications: list[int] = []
        while True:
          for _ in stream_with_trace(graph, graph_input, trace_path, config=config):
            pass
          if pending_notifications:
            with db.open_db(db_path) as conn:
              db.ack_main_notifications(conn, pending_notifications)
            pending_notifications.clear()
          update, pending_notifications = _wait_for_main_event(db_path, args.poll_s, api_key)
          if update is None:
            return 0
          graph_input = {"messages": [HumanMessage(content=update)]}
    except Exception as exc:  # noqa: BLE001
      status = "failed"
      error = f"{type(exc).__name__}: {exc}"
      with db.open_db(db_path) as conn, conn:
        db.notify_main(conn, agent_id=None, run_id=None, kind="main_failed", table_name="agent_sessions", row={"error": error, "trace_ref": str(trace_path)})
      return 1
    finally:
      with db.open_db(db_path) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status=status)

  agent_id = int(args.agent_id)
  db_path = Path(args.db_path).resolve()
  trace_path = Path(args.trace_path).resolve()
  session_name = f"perferox-agent-{agent_id}"
  registry = SessionRegistry()
  api_key = read_cloud_key(args.cloud_key_file)
  provider = cloud_provider(api_key)
  # Expose only the selected credential to the provider CLI subprocess.
  for name in ("LAMBDA_API_KEY", "RUNPOD_API_KEY"):
    os.environ.pop(name, None)
  os.environ["LAMBDA_API_KEY" if provider == "lambda" else "RUNPOD_API_KEY"] = api_key
  status = "exited"
  exit_code = 0
  try:
    with db.open_db(db_path) as conn:
      db.init_db(conn)
      db.record_agent_session(conn, session_name=session_name, role="subagent", agent_id=agent_id, trace_ref=str(trace_path))
      if conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (session_name,)).fetchone()["status"] == "ending":
        return 0
    attempt_cap = int(args.attempt_cap)
    create_prompt = LAMBDA_CREATE_POD_SYSTEM_PROMPT if provider == "lambda" else CREATE_POD_SYSTEM_PROMPT
    checkpoint_path = db_path.with_suffix(".checkpoints.sqlite")
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
      checkpointer.setup()
      graph = build_subagent_graph(
        build_chat_model(role="subagent"), agent_id, registry, db_path, args.repository, args.commit,
        create_pod_prompt=create_prompt,
        attempt_cap=attempt_cap,
        trace_ref=str(trace_path),
        create_pod_tools=(provider_cli(provider, db_path, agent_id),),
        checkpointer=checkpointer,
      )
      config = {"configurable": {"thread_id": f"agent-{agent_id}"}}
      state = {"agent_id": agent_id, "objective": Path(args.goal_file).read_text(encoding="utf-8"), "messages": []}
      graph_input = None if checkpointer.get_tuple(config) else state
      for _ in stream_with_trace(graph, graph_input, trace_path, config=config, agent_id=agent_id):
        pass
  except Exception as exc:  # noqa: BLE001
    status = "failed"
    exit_code = 1
    error = f"{type(exc).__name__}: {exc}"
    with db.open_db(db_path) as conn, conn:
      db.fail_unfinished_runs(conn, agent_id, error)
      db.notify_main(conn, agent_id=agent_id, run_id=None, kind="subagent_failed", table_name="agent_sessions", row={"agent_id": agent_id, "error": error, "trace_ref": str(trace_path)})
  finally:
    registry.close(f"agent-{agent_id}")
    cleanup_errors = cleanup_cloud_resources(db_path, agent_id, api_key)
    if cleanup_errors:
      status = "failed"
      exit_code = 1
      with db.open_db(db_path) as conn, conn:
        db.notify_main(conn, agent_id=agent_id, run_id=None, kind="cleanup_failed", table_name="cloud_resources", row={"agent_id": agent_id, "error": "\n".join(cleanup_errors)})
    with db.open_db(db_path) as conn:
      unfinished = db.fail_unfinished_runs(conn, agent_id, "worker exited before experiment logging")
      if unfinished:
        status = "failed"
        exit_code = 1
      if db.finish_agent_session(conn, session_name=session_name, status=status):
        db.append_explorer_state(conn, agent_id=agent_id, line=f"agent-{agent_id} tmux {status}; trace {trace_path.name}")
  return exit_code


def _wait_for_main_event(db_path: Path, poll_s: float, api_key: str | None = None) -> tuple[str | None, list[int]]:
  """Wait for a durable wakeup, returning notification IDs for later ack."""
  previous_running = None
  while True:
    with db.open_db(db_path) as conn:
      main_row = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()
      missing_line = ""
      for row in reconcile_tmux_sessions(conn, role="subagent"):
        missing_line = f"agent-{row['agent_id']} tmux missing; trace {Path(row['trace_ref']).name}"
        db.append_explorer_state(conn, agent_id=row["agent_id"], line=missing_line)
        cleanup_errors = cleanup_cloud_resources(db_path, int(row["agent_id"]), api_key)
        if cleanup_errors:
          db.append_explorer_state(conn, agent_id=row["agent_id"], line=f"agent-{row['agent_id']} resource cleanup failed")
      active_rows = conn.execute("SELECT session_name FROM agent_sessions WHERE status IN ('running', 'ending') AND role = 'subagent'").fetchall()
      ending = main_row is not None and main_row["status"] == "ending"
      if ending and not active_rows:
        return None, []
      notifications = [] if ending else db.read_main_notifications(conn)
      if notifications:
        lines = ["Subagent SQLite write notifications:"]
        for row in notifications:
          prefix = "ANOMALY " if row["kind"] == "anomaly_logged" else ""
          lines.append(
            f"{prefix}notification_id={row['notification_id']} kind={row['kind']} "
            f"agent_id={row['agent_id']} run_id={row['run_id']} table={row['table_name']}"
          )
          lines.append(row["row_json"])
        return "\n".join(lines), [int(row["notification_id"]) for row in notifications]
      if missing_line:
        return f"Tmux session update:\n{missing_line}", []
    if ending:
      time.sleep(poll_s)
      continue
    running = {row["session_name"] for row in active_rows}
    if previous_running is not None and running != previous_running:
      update = "Tmux session update:\nsubagent tmux sessions changed" if running else "Tmux session update:\nall subagent tmux sessions completed"
      return update, []
    if not running:
      return None, []
    previous_running = running
    time.sleep(poll_s)


if __name__ == "__main__":
  raise SystemExit(main())
