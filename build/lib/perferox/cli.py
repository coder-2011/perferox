"""Rich command-line entry point for Perferox."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from perferox import db
from perferox.auth import chatgpt_auth_ready, cloud_provider, ensure_chatgpt_auth, modal_cloud_key

CONSOLE = Console()
ERROR_CONSOLE = Console(stderr=True)
BRAND = "#fabd2f"
DIM = "#928374"
STATUS_STYLES = {"running": "green", "ok": "green", "ending": "yellow", "warn": "yellow", "failed": "red", "fail": "red", "exited": DIM, "missing": "red", "idle": DIM}


def main(argv: list[str] | None = None) -> int:
  """Route the TUI, agent lifecycle, status, logs, login, and diagnostics."""
  parser = argparse.ArgumentParser(prog="perferox", description="Agentic benchmark explorer for SGLang")
  parser.add_argument("--cwd", type=Path, default=Path("."), help="repository/root directory")
  parser.add_argument("--db-path", type=Path, default=Path("perferox.sqlite"), help="SQLite state path")
  parser.add_argument("--trace-dir", type=Path, default=Path("traces"), help="trace directory")
  parser.add_argument("--version", action="store_true", help="show the installed version and exit")
  subparsers = parser.add_subparsers(dest="command")
  run_parser = subparsers.add_parser("run", help="start the main graph without opening the TUI")
  run_parser.add_argument("objective", nargs="+", help="objective for the main agent")
  run_parser.add_argument("--provider", choices=("runpod", "lambda", "modal"), help="cloud provider; select Modal explicitly because it has no single-key prefix")
  subparsers.add_parser("status", help="show comprehensive persisted run status")
  subparsers.add_parser("login", help="authenticate with ChatGPT OAuth")
  logs_parser = subparsers.add_parser("logs", help="show recent SQLite and trace activity")
  logs_parser.add_argument("-n", "--limit", type=int, default=30, help="maximum activity lines (default: 30)")
  subparsers.add_parser("doctor", help="check local requirements without cloud calls")
  subparsers.add_parser("end", help="request a soft stop without opening the TUI")
  args = parser.parse_args(argv)

  if args.version:
    CONSOLE.print(Text.assemble(("perferox", f"bold {BRAND}"), (f" {version('perferox')}", "bold white")))
    return 0

  cwd = args.cwd.resolve()
  db_path = (cwd / args.db_path).resolve()
  trace_dir = (cwd / args.trace_dir).resolve()

  if args.command is None:
    from perferox.tui import PerferoxTUI

    PerferoxTUI(cwd=cwd, db_path=db_path, trace_dir=trace_dir).run()
    return 0
  if args.command == "login":
    return _login()
  if args.command == "status":
    return _status(db_path)
  if args.command == "logs":
    return _logs(db_path, args.limit)
  if args.command == "doctor":
    return _doctor(cwd, db_path)
  if args.command == "run":
    return _run(args, cwd, db_path, trace_dir)

  try:
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      stopped = db.request_soft_stop(conn)
      db.append_explorer_state(conn, agent_id=None, line="soft stop requested from CLI")
  except (OSError, sqlite3.Error) as exc:
    return _error(f"soft stop failed: {exc}")
  style = "yellow" if stopped else DIM
  message = f"soft stop requested for {stopped} running session(s)" if stopped else "no running sessions"
  CONSOLE.print(Panel.fit(Text(message, style=style), title="[bold]End[/]", border_style=style))
  return 0


def _login() -> int:
  """Report the result of the auth-owned ChatGPT login workflow."""
  try:
    token_saved = ensure_chatgpt_auth()
  except Exception as exc:  # noqa: BLE001
    return _error(f"login failed: {type(exc).__name__}: {exc}")
  message = "token saved" if token_saved else "ChatGPT OAuth is ready"
  CONSOLE.print(Panel.fit(f"[green]authenticated[/] · {message}", title="[bold]Login[/]", border_style="green"))
  return 0


def _run(args: argparse.Namespace, cwd: Path, db_path: Path, trace_dir: Path) -> int:
  """Validate credentials and launch the tmux-wrapped main agent."""
  if not chatgpt_auth_ready():
    return _error("ChatGPT OAuth is missing; run `perferox login` first")
  from perferox.process_host import main as run_agent

  objective = " ".join(args.objective)
  selected = args.provider or Prompt.ask("Cloud provider", choices=("runpod", "lambda", "modal"), console=CONSOLE)
  if selected == "modal":
    try:
      api_key = modal_cloud_key()
    except ValueError as exc:
      return _error(str(exc))
  else:
    env_name = "LAMBDA_API_KEY" if selected == "lambda" else "RUNPOD_API_KEY"
    api_key = os.environ.get(env_name)
    if not api_key:
      api_key = Prompt.ask(f"{selected.title()} API key", password=True, console=CONSOLE)
  try:
    provider = cloud_provider(api_key)
  except ValueError as exc:
    return _error(str(exc))
  if provider != selected:
    CONSOLE.print(f"[yellow]Using {provider} based on the key prefix.[/]")
  return run_agent(
    [
      "launch-main", "--db-path", str(db_path), "--trace-dir", str(trace_dir),
      "--objective", objective, "--cwd", str(cwd),
    ],
    cloud_api_key=api_key,
  )


def _status(db_path: Path) -> int:
  """Render comprehensive persisted state through bounded Rich tables."""
  from perferox.status import read_dashboard

  try:
    snapshot = read_dashboard(db_path, trace_limit=10)
  except (OSError, sqlite3.Error) as exc:
    return _error(f"status failed: {exc}")
  active_agents = sum(1 for session in snapshot.sessions if session["role"] == "subagent" and session["status"] in {"running", "ending"})
  summary = Table.grid(expand=True)
  summary.add_row(Text.assemble(("● ", STATUS_STYLES.get(snapshot.main_status, "white")), (f"main {snapshot.main_status}", "bold"), f"  ·  {active_agents} active subagent(s)"))
  summary.add_row(f"{snapshot.runs} runs  ·  {snapshot.running_runs} running  ·  {snapshot.succeeded_runs} ok  ·  {snapshot.failed_runs} failed  ·  {snapshot.experiments} experiments  ·  {snapshot.anomaly_count} anomalies")
  summary.add_row(Text.assemble(("database  ", DIM), str(db_path)))
  CONSOLE.print(Panel(summary, title=f"[bold {BRAND}]perferox[/] status", border_style=BRAND))

  sessions = Table("State", "Role", "Agent", "Runs", "OK", "Failed", "Trace", title="Sessions", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for session in snapshot.sessions:
    state = str(session["status"])
    sessions.add_row(
      Text(state, style=STATUS_STYLES.get(state, "white")),
      str(session["role"]),
      "—" if session["agent_id"] is None else str(session["agent_id"]),
      str(session["run_count"] or 0),
      str(session["succeeded_runs"] or 0),
      str(session["failed_runs"] or 0),
      Path(str(session["trace_ref"])).name if session.get("trace_ref") else "—",
    )
  if not snapshot.sessions:
    sessions.add_row(Text("idle", style=DIM), "—", "—", "0", "0", "0", "—")
  CONSOLE.print(sessions)

  runs = Table("Run", "State", "Started", "Intent / command", title="Recent runs", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for run in snapshot.recent_runs:
    state = str(run["status"])
    runs.add_row(f"{run['agent_id']}/{run['run_id']}", Text(state, style=STATUS_STYLES[state]), str(run["started_at"]).replace("T", " ")[:19], Text(str(run["label"] or "—")))
  if not snapshot.recent_runs:
    runs.add_row("—", Text("none", style=DIM), "—", "No benchmark runs recorded")
  CONSOLE.print(runs)

  anomalies = Table("ID", "Run", "Date", "Summary", title="Recent anomalies", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for anomaly in snapshot.anomalies:
    anomalies.add_row(f"ANM-{anomaly['anomaly_id']:04d}", f"{anomaly['agent_id']}/{anomaly['run_id']}", str(anomaly["date"]).replace("T", " ")[:19], Text(str(anomaly["summary"]), style="red"))
  if not snapshot.anomalies:
    anomalies.add_row("—", "—", "—", Text("No anomalies logged", style=DIM))
  CONSOLE.print(anomalies)

  activity = Text("\n".join(snapshot.trace_lines) if snapshot.trace_lines else "No activity recorded", style=DIM)
  CONSOLE.print(Panel(activity, title="Recent activity", border_style=DIM))
  return 0


def _logs(db_path: Path, limit: int) -> int:
  """Render a bounded persisted activity tail without importing Textual."""
  if limit < 1:
    return _error("--limit must be at least 1")
  from perferox.status import read_activity

  try:
    lines = read_activity(db_path, limit)
  except (OSError, sqlite3.Error) as exc:
    return _error(f"logs failed: {exc}")
  content = Text("\n".join(lines) if lines else "No activity recorded", style=DIM)
  CONSOLE.print(Panel(content, title=f"[bold {BRAND}]perferox[/] logs · last {limit}", border_style=BRAND))
  return 0


def _doctor(cwd: Path, db_path: Path) -> int:
  """Check local launch requirements without model or cloud API calls."""
  uv = shutil.which("uv")
  tmux = shutil.which("tmux")
  authenticated = chatgpt_auth_ready()
  checks = [
    ("workspace", "ok" if cwd.is_dir() else "fail", str(cwd)),
    ("uv", "ok" if uv else "fail", uv or "not found"),
    ("tmux", "ok" if tmux else "fail", tmux or "not found"),
    ("ChatGPT OAuth", "ok" if authenticated else "fail", "authenticated" if authenticated else "run `perferox login`"),
  ]
  try:
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
    checks.append(("SQLite", "ok", str(db_path)))
  except Exception as exc:  # noqa: BLE001
    checks.append(("SQLite", "fail", f"{type(exc).__name__}: {exc}"))

  configured = []
  for env_name, expected in (("RUNPOD_API_KEY", "runpod"), ("LAMBDA_API_KEY", "lambda")):
    api_key = os.environ.get(env_name)
    if not api_key:
      continue
    try:
      configured.append(expected if cloud_provider(api_key) == expected else f"invalid {env_name}")
    except ValueError:
      configured.append(f"invalid {env_name}")
  modal_config = Path("~/.modal.toml").expanduser()
  if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
    try:
      modal_cloud_key()
      configured.append("modal")
    except ValueError:
      configured.append("invalid Modal tokens")
  elif modal_config.is_file():
    configured.append("modal")
  invalid_cloud = any(item.startswith("invalid") for item in configured)
  cloud_state = "fail" if invalid_cloud else "ok" if configured else "warn"
  cloud_detail = ", ".join(configured) if configured else "RunPod/Lambda prompt on start; Modal uses `modal setup`"
  checks.append(("cloud auth", cloud_state, cloud_detail))
  checkout_ready = (cwd / "sglang" / ".git").is_dir()
  checks.append(("SGLang checkout", "ok" if checkout_ready else "warn", "ready" if checkout_ready else "cloned on first run"))

  table = Table("Check", "State", "Detail", title="Perferox doctor", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for name, state, detail in checks:
    table.add_row(name, Text(state, style=STATUS_STYLES[state]), Text(detail))
  CONSOLE.print(table)
  failures = sum(state == "fail" for _, state, _ in checks)
  CONSOLE.print("[green]ready[/]" if not failures else f"[red]{failures} required check(s) failed[/]")
  return 1 if failures else 0


def _error(message: str) -> int:
  """Print one safe Rich error and return the conventional failure code."""
  ERROR_CONSOLE.print(Text.assemble(("error: ", "bold red"), message))
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
