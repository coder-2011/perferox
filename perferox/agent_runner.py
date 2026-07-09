"""Tiny CLI for tmux-wrapped Perferox agent processes."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import time
from contextlib import closing
from pathlib import Path

from langchain_core.messages import HumanMessage

from perferox import db
from perferox.auth import build_chat_model
from perferox.main_agent import build_main_agent_graph
from perferox.remote import SessionRegistry
from perferox.subagent import build_subagent_graph, stream_with_trace

MAIN_SESSION = "perferox-main"


def main(argv: list[str] | None = None) -> int:
  """Parse the runner command and run the requested whole-agent process."""
  parser = argparse.ArgumentParser(prog="python -m perferox.agent_runner")
  subparsers = parser.add_subparsers(dest="command", required=True)
  for name in ("launch-main", "main"):
    subparser = subparsers.add_parser(name)
    subparser.add_argument("--db-path", required=True)
    subparser.add_argument("--trace-dir", default="traces")
    subparser.add_argument("--objective", required=True)
    subparser.add_argument("--cwd", default=".")
  subparsers.choices["main"].add_argument("--poll-s", type=float, default=5.0)
  subagent = subparsers.add_parser("subagent")
  for name in ("agent-id", "db-path", "trace-path", "goal-file", "attempt-cap"):
    subagent.add_argument(f"--{name}", required=True)
  args = parser.parse_args(argv)

  if args.command == "launch-main":
    tmux = shutil.which("tmux")
    if tmux is None:
      print("tmux is not installed or not on PATH")
      return 1
    cwd = Path(args.cwd).resolve()
    db_path = (cwd / args.db_path).resolve()
    trace_dir = (cwd / args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["uv", "run", "python", "-m", "perferox.agent_runner", "main", "--db-path", str(db_path), "--trace-dir", str(trace_dir), "--objective", args.objective, "--cwd", str(cwd)])
    result = subprocess.run([tmux, "new-session", "-d", "-s", MAIN_SESSION, "-c", str(cwd), "--", "bash", "-lc", command], text=True, capture_output=True, check=False)
    print(f"started {MAIN_SESSION}; attach with: tmux attach -t {MAIN_SESSION}" if result.returncode == 0 else (result.stderr or result.stdout).strip())
    return result.returncode

  if args.command == "main":
    cwd = Path(args.cwd).resolve()
    db_path = (cwd / args.db_path).resolve()
    trace_dir = (cwd / args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{MAIN_SESSION}-{int(time.time())}.jsonl"
    try:
      with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        db.record_agent_session(conn, session_name=MAIN_SESSION, role="main", trace_ref=str(trace_path))
      graph = build_main_agent_graph(build_chat_model(), db_path, cwd=cwd, trace_dir=trace_dir)
      state = {"objective": args.objective, "messages": [HumanMessage(content=args.objective)]}
      while True:
        for event in stream_with_trace(graph, state, trace_path):
          state = _collect_update(state, event)
        update = _wait_for_main_event(db_path, args.poll_s)
        if update is None:
          return 0
        state["messages"] = [*state.get("messages", []), HumanMessage(content=update)]
    finally:
      with closing(db.connect(db_path)) as conn:
        db.finish_agent_session(conn, session_name=MAIN_SESSION, status="exited")

  agent_id = int(args.agent_id)
  db_path = Path(args.db_path).resolve()
  trace_path = Path(args.trace_path).resolve()
  session_name = f"perferox-agent-{agent_id}"
  registry = SessionRegistry()
  try:
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      db.record_agent_session(conn, session_name=session_name, role="subagent", agent_id=agent_id, trace_ref=str(trace_path))
    attempt_cap = int(args.attempt_cap)
    graph = build_subagent_graph(build_chat_model(), agent_id, registry, db_path, attempt_cap=attempt_cap, trace_ref=str(trace_path))
    state = {"agent_id": agent_id, "loop_cap": attempt_cap, "messages": [HumanMessage(content=Path(args.goal_file).read_text(encoding="utf-8"))]}
    for _ in stream_with_trace(graph, state, trace_path):
      pass
    return 0
  finally:
    registry.close(f"agent-{agent_id}")
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      if db.finish_agent_session(conn, session_name=session_name, status="exited"):
        db.append_explorer_state(conn, agent_id=agent_id, line=f"agent-{agent_id} tmux exited; trace {trace_path.name}")


def _collect_update(state: dict, event: object) -> dict:
  """Merge one streamed LangGraph update into a reusable graph state."""
  if not isinstance(event, dict):
    return state
  for update in event.values():
    if not isinstance(update, dict):
      continue
    for key, value in update.items():
      if key == "messages":
        state.setdefault(key, []).extend(value)
      else:
        state[key] = value
  return state


def _wait_for_main_event(db_path: Path, poll_s: float) -> str | None:
  """Wait for a main-agent wakeup or a human End request."""
  previous_running = None
  while True:
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      main_row = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", (MAIN_SESSION,)).fetchone()
      notifications = db.take_main_notifications(conn)
      if notifications:
        lines = ["Subagent SQLite write notifications:"]
        for row in notifications:
          prefix = "ANOMALY " if row["kind"] == "anomaly_logged" else ""
          lines.append(
            f"{prefix}notification_id={row['notification_id']} kind={row['kind']} "
            f"agent_id={row['agent_id']} run_id={row['run_id']} table={row['table_name']}"
          )
          lines.append(row["row_json"])
        return "\n".join(lines)
      tmux = shutil.which("tmux")
      active_query = "SELECT session_name, agent_id, trace_ref FROM agent_sessions WHERE status IN ('running', 'ending') AND role = 'subagent'"
      rows = conn.execute(active_query).fetchall()
      for row in rows:
        alive = tmux and subprocess.run([tmux, "has-session", "-t", row["session_name"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
        if not alive and db.finish_agent_session(conn, session_name=row["session_name"], status="missing"):
          line = f"agent-{row['agent_id']} tmux missing; trace {Path(row['trace_ref']).name}"
          db.append_explorer_state(conn, agent_id=row["agent_id"], line=line)
          return f"Tmux session update:\n{line}"
      if main_row is not None and main_row["status"] == "ending":
        if not rows:
          return None
        time.sleep(poll_s)
        return "End requested. Do not delegate new subagents; wait for active subagents to wrap up, then summarize."
    running = {row["session_name"] for row in rows}
    if previous_running is not None and running != previous_running:
      return "Tmux session update:\nsubagent tmux sessions changed" if running else "Tmux session update:\nall subagent tmux sessions completed"
    if not running:
      time.sleep(poll_s)
      return "No active subagents are running. Continue exploration from the objective and ExplorerState."
    previous_running = running
    time.sleep(poll_s)


if __name__ == "__main__":
  raise SystemExit(main())
