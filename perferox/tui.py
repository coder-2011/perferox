"""Live Textual dashboard for Perferox runs."""

# ruff: noqa: BLE001

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# Textual reads color env vars at import time.
os.environ.pop("NO_COLOR", None)
os.environ.setdefault("TEXTUAL_COLOR_SYSTEM", "truecolor")

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Input, Static

from perferox import db
from perferox.agent_runner import MAIN_SESSION
from perferox.auth import chatgpt_auth_ready, login_chatgpt_oauth

ANOMALY_LIMIT = 8
TRACE_LIMIT = 80
TRACE_TAIL_CHUNK_BYTES = 65536


@dataclass(slots=True)
class DashboardSnapshot:
  """Small immutable view of the SQLite and JSONL state the TUI renders."""

  sessions: list[dict[str, object]]
  anomalies: list[dict[str, object]]
  trace_lines: list[str]
  runs: int
  experiments: int
  main_status: str


def read_dashboard(db_path: str | Path, *, trace_limit: int = TRACE_LIMIT) -> DashboardSnapshot:
  """Read the live run status without consuming main-agent notifications."""
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    sessions = [
      dict(row)
      for row in conn.execute(
        """
        SELECT s.*,
          COUNT(r.run_id) AS run_count,
          SUM(CASE WHEN r.finished_at IS NOT NULL AND r.error = '' THEN 1 ELSE 0 END) AS succeeded_runs,
          SUM(CASE WHEN r.error != '' THEN 1 ELSE 0 END) AS failed_runs
        FROM agent_sessions s
        LEFT JOIN runs r ON r.agent_id = s.agent_id
        GROUP BY s.session_name
        ORDER BY s.role, s.agent_id, s.session_name
        """
      ).fetchall()
    ]
    anomalies = [
      dict(row)
      for row in conn.execute(
        """
        SELECT a.*, r.command
        FROM anomalies a
        LEFT JOIN runs r ON r.agent_id = a.agent_id AND r.run_id = a.run_id
        ORDER BY a.anomaly_id DESC
        LIMIT ?
        """,
        (ANOMALY_LIMIT,),
      ).fetchall()
    ]
    db_events = [
      f"{row['created_at']} explorer: {row['line']}"
      for row in conn.execute("SELECT created_at, line FROM explorer_state_lines ORDER BY line_id DESC LIMIT ?", (trace_limit,)).fetchall()
    ]
    db_events.extend(
      f"{row['created_at']} sqlite {row['kind']}: {_notification_text(row['row_json'])}"
      for row in conn.execute(
        """
        SELECT created_at, kind, row_json
        FROM main_notifications
        ORDER BY notification_id DESC
        LIMIT ?
        """,
        (trace_limit,),
      ).fetchall()
    )
    runs = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    experiments = int(conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

  main_status = next((str(session["status"]) for session in sessions if session["session_name"] == MAIN_SESSION), "idle")
  trace_refs = list(dict.fromkeys(str(session["trace_ref"]) for session in sessions if session.get("trace_ref")))
  trace_lines = [*reversed(db_events), *read_trace_tail(trace_refs, trace_limit)]
  return DashboardSnapshot(
    sessions=sessions,
    anomalies=anomalies,
    trace_lines=trace_lines[-trace_limit:],
    runs=runs,
    experiments=experiments,
    main_status=main_status,
  )


def read_trace_tail(paths: list[str], limit: int) -> list[str]:
  """Return compact trace lines from the newest available JSONL records."""
  lines: deque[str] = deque(maxlen=limit)
  for raw_path in paths:
    path = Path(raw_path)
    if not path.exists():
      continue
    for raw_line in _tail_lines(path, limit):
      lines.append(format_trace_line(path, raw_line))
  return list(lines)


def _tail_lines(path: Path, limit: int) -> list[str]:
  """Read the last trace lines without scanning a long JSONL file."""
  if limit <= 0:
    return []
  chunks = []
  newlines = 0
  with path.open("rb") as file:
    file.seek(0, os.SEEK_END)
    position = file.tell()
    while position > 0 and newlines <= limit:
      read_size = min(TRACE_TAIL_CHUNK_BYTES, position)
      position -= read_size
      file.seek(position)
      chunk = file.read(read_size)
      chunks.append(chunk)
      newlines += chunk.count(b"\n")
  data = b"".join(reversed(chunks))
  return [line.decode("utf-8", "replace") for line in data.splitlines()[-limit:]]


def format_trace_line(path: Path, raw_line: str) -> str:
  """Convert one graph JSONL record into a human-readable TUI line."""
  try:
    record = json.loads(raw_line)
  except json.JSONDecodeError:
    return f"{path.name}: {_short(raw_line.strip(), 300)}"
  ts = str(record.get("ts", ""))
  agent = record.get("agent_id")
  who = "main" if agent is None else f"agent-{agent}"
  text = trace_payload_text(record.get("payload"))
  return _short(f"{ts} {who}: {text}", 500)


def trace_payload_text(payload: object) -> str:
  """Extract the most useful message text from a LangGraph trace payload."""
  message = _find_last_message(payload)
  if isinstance(message, dict):
    content = message.get("content")
    if content:
      return str(content)
    tool_calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get("tool_calls")
    if tool_calls:
      return f"tool calls: {_short(json.dumps(tool_calls, default=str), 220)}"
  return _short(json.dumps(payload, default=str, separators=(",", ":")), 300)


def _notification_text(row_json: str) -> str:
  """Render one SQLite write notification without dumping the whole row."""
  try:
    row = json.loads(row_json)
  except json.JSONDecodeError:
    return _short(row_json, 300)
  parts = []
  for key in ("agent_id", "run_id", "summary", "intent_key", "command", "error"):
    value = row.get(key)
    if value not in (None, ""):
      parts.append(f"{key}={value}")
  return _short(", ".join(parts) or row_json, 300)


def request_end(db_path: str | Path) -> int:
  """Ask every running agent process to stop after its current work."""
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    stopped = db.request_soft_stop(conn)
    db.append_explorer_state(conn, agent_id=None, line="soft stop requested from TUI")
  return stopped


def launch_main(cwd: str | Path, db_path: str | Path, trace_dir: str | Path, objective: str) -> subprocess.CompletedProcess[str]:
  """Start the tmux-wrapped main graph through the existing runner CLI."""
  command = [
    "uv",
    "run",
    "python",
    "-m",
    "perferox.agent_runner",
    "launch-main",
    "--db-path",
    str(Path(db_path).resolve()),
    "--trace-dir",
    str(Path(trace_dir).resolve()),
    "--objective",
    objective,
    "--cwd",
    str(Path(cwd).resolve()),
  ]
  return subprocess.run(command, cwd=Path(cwd), text=True, capture_output=True, check=False)


class PerferoxTUI(App[None]):
  """Render live Perferox state and controls."""

  CSS = """
  Screen { background: #1d2021; color: #d5c4a1; }
  #root { height: 100%; width: 100%; background: #1d2021; }
  #login-screen { height: 1fr; width: 100%; align: center middle; }
  #login-box { width: 44; height: 8; align: center middle; }
  #body { height: 1fr; }
  #left { width: 31; border-right: solid #504945; }
  #main { width: 1fr; min-width: 64; }
  #right { width: 34; border-left: solid #504945; }
  .section-title { height: 2; padding: 0 1; color: #fabd2f; text-style: bold; border-bottom: solid #32302f; }
  .scroll-pane { height: 1fr; padding: 1; scrollbar-color: #504945; scrollbar-background: #1d2021; scrollbar-corner-color: #1d2021; }
  #objective-row { height: 5; padding: 1; background: #282828; border-bottom: solid #504945; }
  #objective { width: 1fr; height: 3; margin-right: 1; background: #1d2021; color: #ebdbb2; border: solid #504945; }
  Button { height: 3; min-width: 10; margin-right: 1; background: #32302f; color: #fabd2f; border: solid #b57614; text-style: bold; }
  Button#end { color: #fb4934; border: solid #fb4934; }
  #footer { height: 1; padding: 0 1; background: #282828; color: #7c6f64; }
  """

  BINDINGS = (("q", "quit", "Quit"),)

  def __init__(
    self,
    *,
    cwd: str | Path = ".",
    db_path: str | Path = "perferox.sqlite",
    trace_dir: str | Path = "traces",
    logged_in: bool | None = None,
  ) -> None:
    """Store paths and initial auth state for the live dashboard."""
    super().__init__()
    self.cwd = Path(cwd).resolve()
    self.db_path = Path(db_path).resolve()
    self.trace_dir = Path(trace_dir).resolve()
    self.logged_in = chatgpt_auth_ready() if logged_in is None else logged_in
    self.login_thread: threading.Thread | None = None

  def compose(self) -> ComposeResult:
    """Build the live dashboard layout."""
    with Vertical(id="root"):
      yield Vertical(Vertical(Button("LOGIN", id="login"), Static("", id="login-status"), id="login-box"), id="login-screen")
      with Horizontal(id="body"):
        yield Vertical(
          Static("STATUS", classes="section-title"),
          Static("", id="counters"),
          Static("SESSIONS", classes="section-title"),
          ScrollableContainer(Static("", id="sessions"), classes="scroll-pane"),
          id="left",
        )
        yield Vertical(
          Horizontal(
            Input(placeholder="Objective", id="objective"),
            Button("START", id="start"),
            Button("END", id="end"),
            id="objective-row",
          ),
          Static("TRACE", classes="section-title"),
          ScrollableContainer(Static("", id="trace-text"), classes="scroll-pane"),
          id="main",
        )
        yield Vertical(
          Static("ANOMALIES", classes="section-title"),
          ScrollableContainer(Static("", id="anomalies-text"), classes="scroll-pane"),
          id="right",
        )
      yield Static("", id="footer")

  def on_mount(self) -> None:
    """Initialize SQLite-backed widgets and start polling."""
    self.trace_dir.mkdir(parents=True, exist_ok=True)
    self._sync_auth_gate()
    self.refresh_dashboard()
    self.set_interval(1.0, self.refresh_dashboard)

  def on_button_pressed(self, event: Button.Pressed) -> None:
    """Route button presses to auth, launch, or soft-stop actions."""
    button_id = event.button.id
    if button_id == "login":
      self._start_login()
    elif button_id == "start":
      self._start_main()
    elif button_id == "end":
      self._end_main()

  def on_input_submitted(self, event: Input.Submitted) -> None:
    """Launch from the objective field when Enter is pressed."""
    if event.input.id == "objective":
      self._start_main()
      event.stop()

  def refresh_dashboard(self) -> None:
    """Re-read SQLite and traces, then update visible state."""
    if not self.logged_in:
      return
    snapshot = read_dashboard(self.db_path)
    self.query_one("#counters", Static).update(_counter_text(snapshot))
    self.query_one("#sessions", Static).update(_session_text(snapshot.sessions))
    self.query_one("#trace-text", Static).update(_trace_text(snapshot.trace_lines))
    self.query_one("#anomalies-text", Static).update(_anomaly_text(snapshot.anomalies))
    footer = {"running": "main graph running", "ending": "soft stop requested; waiting for current work to finish", "exited": "main graph exited"}.get(snapshot.main_status, "idle")
    self.query_one("#footer", Static).update(footer)
    self.query_one("#start", Button).disabled = snapshot.main_status in {"running", "ending"}
    self.query_one("#end", Button).disabled = snapshot.main_status not in {"running", "ending"}

  def _start_main(self) -> None:
    """Launch the tmux-backed main graph for the entered objective."""
    if not self.logged_in:
      return
    objective = self.query_one("#objective", Input).value.strip()
    if not objective:
      self.query_one("#footer", Static).update("enter an objective before starting")
      return
    result = launch_main(self.cwd, self.db_path, self.trace_dir, objective)
    output = (result.stdout or result.stderr).strip()
    self.query_one("#footer", Static).update(output or f"launch exited {result.returncode}")
    self.refresh_dashboard()

  def _end_main(self) -> None:
    """Request a real soft stop through SQLite."""
    stopped = request_end(self.db_path)
    self.query_one("#footer", Static).update(f"soft stop requested for {stopped} running session(s)")
    self.refresh_dashboard()

  def _sync_auth_gate(self) -> None:
    """Hide the working UI until ChatGPT OAuth is available."""
    self.query_one("#login-screen", Vertical).display = not self.logged_in
    self.query_one("#body", Horizontal).display = self.logged_in
    self.query_one("#footer", Static).display = self.logged_in

  def _start_login(self) -> None:
    """Run the blocking OAuth flow in a background thread."""
    if self.login_thread and self.login_thread.is_alive():
      return
    self.query_one("#login", Button).disabled = True
    self.query_one("#login-status", Static).update("opening browser")
    self.login_thread = threading.Thread(target=self._login_worker, daemon=True)
    self.login_thread.start()

  def _login_worker(self) -> None:
    """Complete OAuth and return the result to Textual's thread."""
    try:
      login_chatgpt_oauth()
    except Exception as exc:
      self.call_from_thread(self._finish_login, False, f"{type(exc).__name__}: {exc}")
      return
    self.call_from_thread(self._finish_login, True, "")

  def _finish_login(self, logged_in: bool, error: str) -> None:
    """Apply the completed login result to the dashboard."""
    self.logged_in = logged_in and chatgpt_auth_ready()
    if self.logged_in:
      self._sync_auth_gate()
      self.refresh_dashboard()
      return
    self.query_one("#login", Button).disabled = False
    self.query_one("#login-status", Static).update(escape(error) if error else "login failed")


def _counter_text(snapshot: DashboardSnapshot) -> str:
  """Render compact live counters."""
  active = sum(1 for session in snapshot.sessions if session["status"] in {"running", "ending"} and session["role"] == "subagent")
  return (
    f"\nmain: {snapshot.main_status}\n"
    f"subagents: {active} active\n"
    f"runs: {snapshot.runs}\n"
    f"experiments: {snapshot.experiments}\n"
    f"anomalies: {len(snapshot.anomalies)} recent\n"
  )


def _session_text(sessions: list[dict[str, object]]) -> str:
  """Render the tmux-backed agent session list."""
  if not sessions:
    return "no sessions"
  lines = []
  for session in sessions:
    agent = "" if session["agent_id"] is None else f" agent-{session['agent_id']}"
    trace = Path(str(session["trace_ref"])).name if session.get("trace_ref") else "no-trace"
    counts = f"{session['run_count']} runs, {session['succeeded_runs']} ok, {session['failed_runs']} failed"
    lines.append(f"{session['status']} {session['role']}{agent}\n  {session['session_name']}\n  {counts}\n  {trace}")
  return "\n\n".join(lines)


def _trace_text(lines: list[str]) -> str:
  """Render trace lines or a clear empty state."""
  return "\n\n".join(escape(line) for line in lines) if lines else "no trace records yet"


def _anomaly_text(anomalies: list[dict[str, object]]) -> str:
  """Render recent anomaly rows."""
  if not anomalies:
    return "no anomalies logged"
  lines = []
  for anomaly in anomalies:
    header = f"ANM-{anomaly['anomaly_id']:04d} agent-{anomaly['agent_id']} run-{anomaly['run_id']}"
    lines.append(f"[#fb4934]{header}[/]\n{escape(str(anomaly['summary']))}\n[#7c6f64]{escape(str(anomaly.get('command') or ''))}[/]")
  return "\n\n".join(lines)


def _find_last_message(value: object) -> object | None:
  """Find the deepest final LangChain message-shaped dict in a trace payload."""
  if isinstance(value, dict):
    messages = value.get("messages")
    if isinstance(messages, list) and messages:
      return messages[-1]
    found = None
    for child in value.values():
      child_found = _find_last_message(child)
      if child_found is not None:
        found = child_found
    return found
  if isinstance(value, list):
    found = None
    for child in value:
      child_found = _find_last_message(child)
      if child_found is not None:
        found = child_found
    return found
  return None


def _short(text: str, limit: int) -> str:
  """Collapse whitespace and cap one visible line."""
  compact = " ".join(text.split())
  return compact if len(compact) <= limit else compact[:limit - 1].rstrip() + "..."


if __name__ == "__main__":
  PerferoxTUI().run()
