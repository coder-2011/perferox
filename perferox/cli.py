"""Small command-line entry point for Perferox."""

from __future__ import annotations

import argparse
import os
from contextlib import closing
from pathlib import Path

from rich.prompt import Prompt

from perferox import db
from perferox.auth import cloud_provider


def main(argv: list[str] | None = None) -> int:
  """Open the TUI, launch the main agent, or request a soft stop."""
  parser = argparse.ArgumentParser(prog="perferox")
  parser.add_argument("--cwd", type=Path, default=Path("."), help="repository/root directory")
  parser.add_argument("--db-path", type=Path, default=Path("perferox.sqlite"), help="SQLite state path")
  parser.add_argument("--trace-dir", type=Path, default=Path("traces"), help="trace directory")
  subparsers = parser.add_subparsers(dest="command")
  run_parser = subparsers.add_parser("run", help="start the main graph without opening the TUI")
  run_parser.add_argument("objective", nargs="+", help="objective for the main agent")
  run_parser.add_argument("--provider", choices=("runpod", "lambda"), help="cloud provider; inferred from a recognized key prefix")
  subparsers.add_parser("status", help="print persisted run status")
  subparsers.add_parser("end", help="request a soft stop without opening the TUI")
  args = parser.parse_args(argv)

  cwd = args.cwd.resolve()
  db_path = (cwd / args.db_path).resolve()
  trace_dir = (cwd / args.trace_dir).resolve()

  if args.command is None:
    from perferox.tui import PerferoxTUI

    PerferoxTUI(cwd=cwd, db_path=db_path, trace_dir=trace_dir).run()
    return 0
  if args.command == "run":
    from perferox.agent_runner import main as run_agent

    objective = " ".join(args.objective)
    selected = args.provider or Prompt.ask("Cloud provider", choices=("runpod", "lambda"))
    env_name = "LAMBDA_API_KEY" if selected == "lambda" else "RUNPOD_API_KEY"
    api_key = os.environ.get(env_name) or Prompt.ask(f"{selected.title()} API key", password=True)
    provider = cloud_provider(api_key)
    if provider != selected:
      print(f"Using {provider} based on the key prefix.")
    return run_agent(
      [
        "launch-main", "--db-path", str(db_path), "--trace-dir", str(trace_dir),
        "--objective", objective, "--cwd", str(cwd),
      ],
      cloud_api_key=api_key,
    )
  if args.command == "status":
    from perferox.tui import read_dashboard

    snapshot = read_dashboard(db_path, trace_limit=10)
    active = sum(1 for session in snapshot.sessions if session["status"] in {"running", "ending"} and session["role"] == "subagent")
    print("Perferox status")
    print(f"  main: {snapshot.main_status}")
    print(f"  subagents: {active} active")
    print(f"  runs: {snapshot.runs}  experiments: {snapshot.experiments}  anomalies: {len(snapshot.anomalies)}")
    print("\nsessions")
    if not snapshot.sessions:
      print("  none")
    for session in snapshot.sessions:
      agent = "" if session["agent_id"] is None else f" agent-{session['agent_id']}"
      trace = Path(str(session["trace_ref"])).name if session.get("trace_ref") else "no-trace"
      counts = f"{session['run_count'] or 0} runs, {session['succeeded_runs'] or 0} ok, {session['failed_runs'] or 0} failed"
      print(f"  {session['status']} {session['role']}{agent}: {session['session_name']}")
      print(f"    {counts}; trace: {trace}")
    print("\nanomalies")
    if not snapshot.anomalies:
      print("  none")
    for anomaly in snapshot.anomalies:
      print(f"  agent-{anomaly['agent_id']} run-{anomaly['run_id']}: {anomaly['summary']}")
    print("\ntrace")
    if not snapshot.trace_lines:
      print("  no trace records yet")
    for line in snapshot.trace_lines:
      print(f"  {line}")
    return 0
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    stopped = db.request_soft_stop(conn)
    db.append_explorer_state(conn, agent_id=None, line="soft stop requested from CLI")
  print(f"soft stop requested for {stopped} running session(s)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
