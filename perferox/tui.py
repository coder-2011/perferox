"""Live Textual dashboard for Perferox runs."""

# ruff: noqa: BLE001

from __future__ import annotations

import os
import subprocess
import threading
from contextlib import closing
from pathlib import Path

# Textual reads color env vars at import time.
os.environ.pop("NO_COLOR", None)
os.environ.setdefault("TEXTUAL_COLOR_SYSTEM", "truecolor")

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Input, Select, Static

from perferox import db
from perferox.auth import chatgpt_auth_ready, cloud_provider, login_chatgpt_oauth
from perferox.status import DashboardSnapshot, read_dashboard


def request_end(db_path: str | Path) -> int:
  """Ask every running agent process to stop after its current work."""
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    stopped = db.request_soft_stop(conn)
    db.append_explorer_state(conn, agent_id=None, line="soft stop requested from TUI")
  return stopped


def launch_main(
  cwd: str | Path,
  db_path: str | Path,
  trace_dir: str | Path,
  objective: str,
  cloud_api_key: str,
) -> subprocess.CompletedProcess[str]:
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
  return subprocess.run(command, cwd=Path(cwd), input=cloud_api_key, text=True, capture_output=True, check=False)


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
  #cloud-provider { width: 16; margin-right: 1; }
  #cloud-key { width: 24; margin-right: 1; }
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
            Select((("RunPod", "runpod"), ("Lambda", "lambda")), prompt="Provider", id="cloud-provider"),
            Input(placeholder="API key", password=True, id="cloud-key"),
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
    if event.input.id in ("cloud-key", "objective"):
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
    api_key = self.query_one("#cloud-key", Input).value.strip()
    try:
      provider = cloud_provider(api_key)
    except ValueError as exc:
      self.query_one("#footer", Static).update(escape(str(exc)))
      return
    objective = self.query_one("#objective", Input).value.strip()
    if not objective:
      self.query_one("#footer", Static).update("enter an objective before starting")
      return
    selected = self.query_one("#cloud-provider", Select).value
    correction = f"using {provider} based on key; " if selected is not Select.BLANK and selected != provider else ""
    result = launch_main(self.cwd, self.db_path, self.trace_dir, objective, api_key)
    self.query_one("#cloud-key", Input).value = ""
    output = (result.stdout or result.stderr).strip()
    self.query_one("#footer", Static).update(correction + (output or f"launch exited {result.returncode}"))
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
    f"anomalies: {snapshot.anomaly_count}\n"
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


if __name__ == "__main__":
  PerferoxTUI().run()
