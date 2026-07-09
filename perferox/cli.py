"""Small command-line entry point for Perferox."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from perferox import db
from perferox.agent_runner import main as run_agent
from perferox.tui import PerferoxTUI


def main(argv: list[str] | None = None) -> int:
  """Open the TUI, launch the main agent, or request a soft stop."""
  parser = argparse.ArgumentParser(prog="perferox")
  parser.add_argument("--cwd", type=Path, default=Path("."), help="repository/root directory")
  parser.add_argument("--db-path", type=Path, default=Path("perferox.sqlite"), help="SQLite state path")
  parser.add_argument("--trace-dir", type=Path, default=Path("traces"), help="trace directory")
  subparsers = parser.add_subparsers(dest="command")
  run_parser = subparsers.add_parser("run", help="start the main graph without opening the TUI")
  run_parser.add_argument("objective", nargs="+", help="objective for the main agent")
  subparsers.add_parser("end", help="request a soft stop without opening the TUI")
  args = parser.parse_args(argv)

  cwd = args.cwd.resolve()
  db_path = (cwd / args.db_path).resolve()
  trace_dir = (cwd / args.trace_dir).resolve()

  if args.command is None:
    PerferoxTUI(cwd=cwd, db_path=db_path, trace_dir=trace_dir).run()
    return 0
  if args.command == "run":
    objective = " ".join(args.objective)
    return run_agent(["launch-main", "--db-path", str(db_path), "--trace-dir", str(trace_dir), "--objective", objective, "--cwd", str(cwd)])
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    stopped = db.request_soft_stop(conn)
    db.append_explorer_state(conn, agent_id=None, line="soft stop requested from CLI")
  print(f"soft stop requested for {stopped} running session(s)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
