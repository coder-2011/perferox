"""Host tmux-wrapped Perferox agent processes."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.text import Text

from perferox import db
from perferox.auth import build_chat_model, cloud_provider, read_cloud_key, write_cloud_key
from perferox.main_agent import build_main_agent_graph
from perferox.prompts import CREATE_POD_SYSTEM_PROMPT, LAMBDA_CREATE_POD_SYSTEM_PROMPT
from perferox.remote import SessionRegistry
from perferox.status import refresh_sessions
from perferox.subagent import build_subagent_graph, stream_with_trace

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
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      # Register before tmux starts so an immediate End request cannot be lost.
      db.finish_agent_session(conn, session_name=MAIN_SESSION, status="missing")
      db.record_agent_session(conn, session_name=MAIN_SESSION, role="main")
    command = shlex.join([
      "uv", "run", "python", "-m", "perferox.process_host", "main",
      "--db-path", str(db_path), "--trace-dir", str(trace_dir),
      "--objective", args.objective, "--cwd", str(cwd),
      "--cloud-key-file", str(key_path),
    ])
    try:
      result = subprocess.run([tmux, "new-session", "-d", "-s", MAIN_SESSION, "-c", str(cwd), "--", "bash", "-lc", command], text=True, capture_output=True, check=False)
    except OSError:
      # Delete a secret handoff that tmux never delivered.
      key_path.unlink(missing_ok=True)
      with closing(db.connect(db_path)) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status="missing")
      raise
    if result.returncode != 0:
      key_path.unlink(missing_ok=True)
      with closing(db.connect(db_path)) as conn:
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
    trace_path = trace_dir / f"{MAIN_SESSION}-{int(time.time())}.jsonl"
    workspace = cwd / "sglang"
    try:
      with closing(db.connect(db_path)) as conn, conn:
        db.init_db(conn)
        db.record_agent_session(conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path))
        # A prior coordinator may have died after reserving but before launching tmux.
        conn.execute("UPDATE agent_sessions SET status = 'missing' WHERE role = 'subagent' AND status IN ('running', 'ending') AND trace_ref = ''")
        session = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()
        if session["status"] == "ending":
          return 0
      # Preserve the shared checkout so agents can keep branches, commits, and edits.
      if not (workspace / ".git").is_dir():
        subprocess.run(["git", "clone", "https://github.com/sgl-project/sglang.git", str(workspace)], check=True)
      graph = build_main_agent_graph(
        build_chat_model(), db_path,
        cloud_provider=provider, cloud_api_key=api_key,
        cwd=workspace, runtime_cwd=cwd, trace_dir=trace_dir,
      )
      state = {"objective": args.objective, "messages": []}
      while True:
        for event in stream_with_trace(graph, state, trace_path):
          _collect_update(state, event)
        update = _wait_for_main_event(db_path, args.poll_s)
        if update is None:
          return 0
        state.setdefault("messages", []).append(HumanMessage(content=update))
    finally:
      with closing(db.connect(db_path)) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status="exited")

  agent_id = int(args.agent_id)
  db_path = Path(args.db_path).resolve()
  trace_path = Path(args.trace_path).resolve()
  session_name = f"perferox-agent-{agent_id}"
  registry = SessionRegistry()
  api_key = read_cloud_key(args.cloud_key_file)
  provider = cloud_provider(api_key)
  # Expose only the selected CLI credential to local_terminal.
  for name in ("LAMBDA_API_KEY", "RUNPOD_API_KEY"):
    os.environ.pop(name, None)
  os.environ["LAMBDA_API_KEY" if provider == "lambda" else "RUNPOD_API_KEY"] = api_key
  try:
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      db.record_agent_session(conn, session_name=session_name, role="subagent", agent_id=agent_id, trace_ref=str(trace_path))
      if conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (session_name,)).fetchone()["status"] == "ending":
        return 0
    attempt_cap = int(args.attempt_cap)
    create_prompt = LAMBDA_CREATE_POD_SYSTEM_PROMPT if provider == "lambda" else CREATE_POD_SYSTEM_PROMPT
    graph = build_subagent_graph(
      build_chat_model(), agent_id, registry, db_path, args.repository, args.commit,
      create_pod_prompt=create_prompt, attempt_cap=attempt_cap, trace_ref=str(trace_path),
    )
    state = {"agent_id": agent_id, "objective": Path(args.goal_file).read_text(encoding="utf-8"), "messages": []}
    for _ in stream_with_trace(graph, state, trace_path):
      pass
    return 0
  finally:
    registry.close(f"agent-{agent_id}")
    with closing(db.connect(db_path)) as conn:
      if db.finish_agent_session(conn, session_name=session_name, status="exited"):
        db.append_explorer_state(conn, agent_id=agent_id, line=f"agent-{agent_id} tmux exited; trace {trace_path.name}")


def _collect_update(state: dict, event: object) -> None:
  """Merge one streamed LangGraph update into a reusable graph state."""
  for update in event.values() if isinstance(event, dict) else ():
    if not isinstance(update, dict):
      continue
    if "messages" in update:
      state.setdefault("messages", []).extend(update["messages"])
    state.update((key, value) for key, value in update.items() if key != "messages")


def _wait_for_main_event(db_path: Path, poll_s: float) -> str | None:
  """Wait for a main-agent wakeup or a human End request."""
  previous_running = None
  while True:
    with closing(db.connect(db_path)) as conn:
      main_row = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()
      notifications = db.take_main_notifications(conn)
      missing = refresh_sessions(conn)
      active_rows = conn.execute("SELECT session_name, agent_id, trace_ref FROM agent_sessions WHERE status IN ('running', 'ending') AND role = 'subagent'").fetchall()
      ending = main_row is not None and main_row["status"] == "ending"
      if ending and not active_rows:
        return None
      if notifications and not ending:
        return "Subagent SQLite write notifications:\n" + "\n".join(
          f"{'ANOMALY ' if row['kind'] == 'anomaly_logged' else ''}notification_id={row['notification_id']} "
          f"kind={row['kind']} agent_id={row['agent_id']} run_id={row['run_id']} table={row['table_name']}\n{row['row_json']}"
          for row in notifications
        )
      if missing:
        return "Tmux session update:\n" + "\n".join(missing)
    if ending:
      time.sleep(poll_s)
      continue
    running = {row["session_name"] for row in active_rows}
    if previous_running is not None and running != previous_running:
      return "Tmux session update:\nsubagent tmux sessions changed" if running else "Tmux session update:\nall subagent tmux sessions completed"
    if not running:
      time.sleep(poll_s)
      return "No active subagents are running. Continue exploration from the objective and ExplorerState."
    previous_running = running
    time.sleep(poll_s)


if __name__ == "__main__":
  raise SystemExit(main())
