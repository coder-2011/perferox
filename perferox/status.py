"""Bounded SQLite and trace reads shared by the CLI and TUI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from perferox import db


@dataclass(slots=True)
class DashboardSnapshot:
  """Store one compact view of persisted sessions, runs, and activity."""

  sessions: list[dict[str, object]]
  recent_runs: list[dict[str, object]]
  anomalies: list[dict[str, object]]
  trace_lines: list[str]
  runs: int
  running_runs: int
  succeeded_runs: int
  failed_runs: int
  experiments: int
  anomaly_count: int
  main_status: str


def read_dashboard(db_path: str | Path, *, trace_limit: int = 80) -> DashboardSnapshot:
  """Read comprehensive status without consuming main-agent notifications."""
  with db.open_db(db_path) as conn:
    db.init_db(conn)
    sessions = [dict(row) for row in conn.execute(
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
    )]
    recent_runs = [dict(row) for row in conn.execute(
        """
        SELECT r.agent_id, r.run_id, r.started_at,
          CASE WHEN r.error != '' THEN 'failed' WHEN r.finished_at IS NOT NULL THEN 'ok' ELSE 'running' END AS status,
          COALESCE(NULLIF(r.intent_key, ''), r.command, '') AS label
        FROM runs r
        ORDER BY r.started_at DESC, r.agent_id DESC, r.run_id DESC
        LIMIT ?
        """,
        (8,),
    )]
    anomalies = [dict(row) for row in conn.execute(
        """
        SELECT a.*, r.command
        FROM anomalies a
        LEFT JOIN runs r ON r.agent_id = a.agent_id AND r.run_id = a.run_id
        ORDER BY a.anomaly_id DESC
        LIMIT ?
        """,
        (8,),
    )]
    counts = dict(conn.execute(
      """
      SELECT
        COUNT(*) AS runs,
        COALESCE(SUM(finished_at IS NULL), 0) AS running_runs,
        COALESCE(SUM(finished_at IS NOT NULL AND error = ''), 0) AS succeeded_runs,
        COALESCE(SUM(error != ''), 0) AS failed_runs,
        (SELECT COUNT(*) FROM experiments) AS experiments,
        (SELECT COUNT(*) FROM anomalies) AS anomaly_count
      FROM runs
      """
    ).fetchone())
    trace_lines = _read_activity(conn, trace_limit)

  main_status = next((str(session["status"]) for session in sessions if session["role"] == "main"), "idle")
  return DashboardSnapshot(sessions, recent_runs, anomalies, trace_lines, main_status=main_status, **counts)


def read_activity(db_path: str | Path, limit: int) -> list[str]:
  """Read only the bounded activity stream needed by `perferox logs`."""
  with db.open_db(db_path) as conn:
    db.init_db(conn)
    return _read_activity(conn, limit)


def _read_activity(conn, limit: int) -> list[str]:
  """Merge recent SQLite events with bounded JSONL trace tails."""
  rows = conn.execute(
    """
    SELECT created_at, kind, payload FROM (
      SELECT created_at, 'explorer' AS kind, line AS payload FROM explorer_state_lines
      UNION ALL
      SELECT created_at, 'sqlite ' || kind AS kind, row_json AS payload FROM main_notifications
    )
    ORDER BY created_at DESC
    LIMIT ?
    """,
    (limit,),
  )
  events = [
    f"{row['created_at']} {row['kind']}: {row['payload'] if row['kind'] == 'explorer' else _notification_text(row['payload'])}"
    for row in rows
  ]
  trace_refs = list(dict.fromkeys(
    row["trace_ref"]
    for row in conn.execute("SELECT trace_ref FROM agent_sessions WHERE trace_ref != '' ORDER BY rowid DESC LIMIT 8")
  ))
  return sorted([*events, *read_trace_tail(trace_refs, limit)])[-limit:]


def read_trace_tail(paths: list[str], limit: int) -> list[str]:
  """Return the newest compact trace lines in timestamp order."""
  lines: list[tuple[str, str]] = []
  for raw_path in paths:
    path = Path(raw_path)
    if not path.exists():
      continue
    for raw_line in _tail_lines(path, limit):
      try:
        timestamp = str(json.loads(raw_line).get("ts", ""))
      except json.JSONDecodeError:
        timestamp = ""
      lines.append((timestamp, format_trace_line(path, raw_line)))
  lines.sort(key=lambda item: item[0])
  return [line for _, line in lines[-limit:]]


def _tail_lines(path: Path, limit: int) -> list[str]:
  """Read the final lines of one trace in bounded backward chunks."""
  if limit <= 0:
    return []
  chunks = []
  newlines = 0
  with path.open("rb") as file:
    file.seek(0, os.SEEK_END)
    position = file.tell()
    while position > 0 and newlines <= limit:
      read_size = min(65536, position)
      position -= read_size
      file.seek(position)
      chunk = file.read(read_size)
      chunks.append(chunk)
      newlines += chunk.count(b"\n")
  data = b"".join(reversed(chunks))
  return [line.decode("utf-8", "replace") for line in data.splitlines()[-limit:]]


def format_trace_line(path: Path, raw_line: str) -> str:
  """Convert one graph JSONL record into a compact human-readable line."""
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
  """Extract the final useful LangChain message from a trace payload."""
  update = next(reversed(payload.values()), {}) if isinstance(payload, dict) else {}
  messages = update.get("messages", ()) if isinstance(update, dict) else ()
  message = messages[-1] if messages else None
  if isinstance(message, dict):
    content = message.get("content")
    if content:
      return str(content)
    tool_calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get("tool_calls")
    if tool_calls:
      return f"tool calls: {_short(json.dumps(tool_calls, default=str), 220)}"
  return _short(json.dumps(payload, default=str, separators=(",", ":")), 300)


def _notification_text(row_json: str) -> str:
  """Render one SQLite notification without dumping its complete row."""
  try:
    row = json.loads(row_json)
  except json.JSONDecodeError:
    return _short(row_json, 300)
  parts = [
    f"{key}={row[key]}"
    for key in ("agent_id", "run_id", "summary", "intent_key", "command", "error")
    if row.get(key) not in (None, "")
  ]
  return _short(", ".join(parts) or row_json, 300)


def _short(text: str, limit: int) -> str:
  """Collapse whitespace and cap one visible line."""
  compact = " ".join(text.split())
  return compact if len(compact) <= limit else compact[:limit - 1].rstrip() + "..."
