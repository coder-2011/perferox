"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_EMBEDDER = None

METRIC_COLUMNS = (
  "request_rps",
  "input_tps",
  "output_tps",
  "ttft_p50_ms",
  "ttft_p99_ms",
  "tpot_p50_ms",
  "tpot_p99_ms",
  "error_rate",
  "cache_hit_rate",
  "peak_gpu_mem_gb",
  "startup_s",
  "warmup_s",
  "accept_length",
  "correctness_score",
)


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
  """Open one SQLite connection for a worker or tool call."""
  if readonly:
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
  else:
    conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA busy_timeout = 5000")
  if readonly:
    conn.execute("PRAGMA query_only = ON")
  else:
    conn.execute("PRAGMA journal_mode = WAL")
  return conn


def init_db(conn: sqlite3.Connection) -> None:
  schema_path = Path(__file__).with_name("init-db.sql")
  conn.executescript(schema_path.read_text(encoding="utf-8"))


def encode_embedding(embedding: Sequence[float]) -> str:
  values = [float(value) for value in embedding]
  return json.dumps(values, separators=(",", ":"))


def embed_intent(intent_key: str) -> list[float]:
  global _EMBEDDER
  if _EMBEDDER is None:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  return list(map(float, _EMBEDDER.encode(intent_key, normalize_embeddings=True)))


def read_explorer_state(conn: sqlite3.Connection) -> list[str]:
  """Return compact ExplorerState lines in insertion order."""
  rows = conn.execute("SELECT line FROM explorer_state_lines ORDER BY line_id").fetchall()
  return [str(row["line"]) for row in rows]


def append_explorer_state(conn: sqlite3.Connection, *, agent_id: int | None, line: str) -> int:
  """Append one compact ExplorerState line."""
  created_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    cursor = conn.execute(
      """
      INSERT INTO explorer_state_lines(agent_id, created_at, line)
      VALUES (?, ?, ?)
      """,
      (agent_id, created_at, line),
    )
  return int(cursor.lastrowid)


def record_agent_session(conn: sqlite3.Connection, *, session_name: str, role: str, agent_id: int | None = None, trace_ref: str = "") -> None:
  """Record one tmux-wrapped agent process as running."""
  with conn:
    conn.execute(
      """
      INSERT OR REPLACE INTO agent_sessions(session_name, role, agent_id, status, trace_ref)
      VALUES (?, ?, ?, 'running', ?)
      """,
      (session_name, role, agent_id, trace_ref),
    )


def finish_agent_session(conn: sqlite3.Connection, *, session_name: str, status: str) -> bool:
  """Mark a tmux-wrapped agent process as exited or missing."""
  with conn:
    cursor = conn.execute(
      """
      UPDATE agent_sessions
      SET status = ?
      WHERE session_name = ? AND status IN ('running', 'ending')
      """,
      (status, session_name),
    )
  return cursor.rowcount == 1


def request_agent_end(conn: sqlite3.Connection, *, session_name: str) -> bool:
  """Mark a running agent session ending after a human End action."""
  with conn:
    cursor = conn.execute(
      "UPDATE agent_sessions SET status = 'ending' WHERE session_name = ? AND status = 'running'",
      (session_name,),
    )
  return cursor.rowcount == 1


def request_soft_stop(conn: sqlite3.Connection) -> int:
  """Mark every running tmux-wrapped agent session as ending."""
  with conn:
    cursor = conn.execute("UPDATE agent_sessions SET status = 'ending' WHERE status = 'running'")
  return cursor.rowcount


def stop_requested(conn: sqlite3.Connection, *, agent_id: int) -> bool:
  """Return whether the main run or this subagent has been asked to stop."""
  row = conn.execute(
    """
    SELECT 1 FROM agent_sessions
    WHERE status = 'ending'
      AND (session_name = 'perferox-main' OR (role = 'subagent' AND agent_id = ?))
    LIMIT 1
    """,
    (agent_id,),
  ).fetchone()
  return row is not None


def take_main_notifications(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
  """Return unread write notifications and mark them delivered."""
  with conn:
    rows = conn.execute(
      """
      SELECT * FROM main_notifications
      WHERE delivered_at IS NULL
      ORDER BY notification_id
      LIMIT ?
      """,
      (limit,),
    ).fetchall()
    if rows:
      delivered_at = datetime.now(UTC).isoformat(timespec="seconds")
      placeholders = ",".join("?" for _ in rows)
      conn.execute(
        f"UPDATE main_notifications SET delivered_at = ? WHERE notification_id IN ({placeholders})",
        (delivered_at, *(row["notification_id"] for row in rows)),
      )
  return rows


def notify_main(conn: sqlite3.Connection, *, agent_id: int | None, run_id: int | None, kind: str, table_name: str, row: Mapping[str, object]) -> None:
  """Queue one host event for the main agent to inspect."""
  conn.execute(
    """
    INSERT INTO main_notifications(created_at, agent_id, run_id, kind, table_name, row_json)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (
      datetime.now(UTC).isoformat(timespec="seconds"),
      agent_id,
      run_id,
      kind,
      table_name,
      json.dumps(dict(row), separators=(",", ":"), default=str),
    ),
  )


def start_benchmark_run(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  command: str,
  trace_ref: str = "",
  attempt_cap: int | None = None,
) -> int:
  """Assign the next run id and insert the started benchmark row."""
  exact_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
  started_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    conn.execute("BEGIN IMMEDIATE")
    if stop_requested(conn, agent_id=agent_id):
      raise ValueError("stop requested; wrap up")
    if attempt_cap is not None:
      attempts = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE agent_id = ?",
        (agent_id,),
      ).fetchone()[0]
      if attempts >= attempt_cap:
        raise ValueError(f"attempt cap reached ({attempts}/{attempt_cap}); wrap up")
    row = conn.execute(
      "SELECT COALESCE(MAX(run_id) + 1, 0) AS run_id FROM runs WHERE agent_id = ?",
      (agent_id,),
    ).fetchone()
    run_id = int(row["run_id"])
    conn.execute(
      """
      INSERT INTO runs(agent_id, run_id, started_at, trace_ref, command, exact_hash)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      (agent_id, run_id, started_at, trace_ref, command, exact_hash),
    )
    row = conn.execute("SELECT * FROM runs WHERE agent_id = ? AND run_id = ?", (agent_id, run_id)).fetchone()
    _insert_main_notification(conn, agent_id=agent_id, run_id=run_id, kind="run_started", table_name="runs", row=row)
  return run_id


def mark_run_failed(conn: sqlite3.Connection, *, agent_id: int, run_id: int, error: str) -> None:
  """Mark a started benchmark run as finished with an error."""
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    conn.execute(
      "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND run_id = ?",
      (finished_at, error[:2000], agent_id, run_id),
    )
    row = conn.execute("SELECT * FROM runs WHERE agent_id = ? AND run_id = ?", (agent_id, run_id)).fetchone()
    _insert_main_notification(conn, agent_id=agent_id, run_id=run_id, kind="run_failed", table_name="runs", row=row)


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  intent_key: str,
  metrics: Mapping[str, float | int | None] | None = None,
) -> int:
  """Atomically save benchmark metrics and mark the run successful."""
  metrics = metrics or {}
  unknown = sorted(set(metrics) - set(METRIC_COLUMNS))
  if unknown:
    raise ValueError(f"unknown metric columns: {', '.join(unknown)}")
  normalized_metrics = {}
  for name, value in metrics.items():
    if value is None:
      continue
    if isinstance(value, bool):
      raise TypeError(f"{name} must be a finite number or null")
    try:
      normalized = float(value)
    except (TypeError, ValueError):
      raise TypeError(f"{name} must be a finite number or null") from None
    if name in ("error_rate", "cache_hit_rate") and normalized > 1.0:
      normalized /= 100.0
    if not math.isfinite(normalized) or normalized < 0.0:
      raise ValueError(f"{name} must be finite and >= 0")
    if name in ("error_rate", "cache_hit_rate") and normalized > 1.0:
      raise ValueError(f"{name} must normalize to a 0..1 rate")
    normalized_metrics[name] = normalized

  row = conn.execute(
    """
    SELECT run_id FROM runs
    WHERE agent_id = ? AND finished_at IS NULL AND error = ''
    ORDER BY run_id DESC
    LIMIT 1
    """,
    (agent_id,),
  ).fetchone()
  if row is None:
    raise ValueError(f"no unfinished successful benchmark run for agent_id={agent_id}")
  run_id = int(row["run_id"])

  columns = ", ".join(METRIC_COLUMNS)
  placeholders = ", ".join("?" for _ in METRIC_COLUMNS)
  values = [normalized_metrics.get(column) for column in METRIC_COLUMNS]
  intent_embedding = embed_intent(intent_key)
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")

  with conn:
    cursor = conn.execute(
      "UPDATE runs SET finished_at = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL",
      (finished_at, agent_id, run_id),
    )
    if cursor.rowcount != 1:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    row = conn.execute("SELECT * FROM runs WHERE agent_id = ? AND run_id = ?", (agent_id, run_id)).fetchone()
    _insert_main_notification(conn, agent_id=agent_id, run_id=run_id, kind="run_succeeded", table_name="runs", row=row)
    conn.execute(
      f"""
      INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, {columns})
      VALUES (?, ?, ?, ?, {placeholders})
      """,
      (agent_id, run_id, intent_key, encode_embedding(intent_embedding), *values),
    )
    row = conn.execute("SELECT * FROM experiments WHERE agent_id = ? AND run_id = ?", (agent_id, run_id)).fetchone()
    _insert_main_notification(conn, agent_id=agent_id, run_id=run_id, kind="experiment_logged", table_name="experiments", row=row)
  return run_id


def find_similar_experiments(conn: sqlite3.Connection, intent: str, limit: int = 5) -> list[dict[str, object]]:
  """Return logged experiments closest to an intent embedding."""
  query_embedding = embed_intent(intent)
  metric_columns = ", ".join(f"e.{column}" for column in METRIC_COLUMNS)
  rows = conn.execute(
    f"""
    SELECT e.agent_id, e.run_id, e.intent_key, e.intent_embedding, {metric_columns},
      r.trace_ref, r.command, r.started_at, r.finished_at, r.error
    FROM experiments e
    JOIN runs r ON r.agent_id = e.agent_id AND r.run_id = e.run_id
    """
  ).fetchall()
  scored = []
  for row in rows:
    entry = dict(row)
    embedding = json.loads(entry.pop("intent_embedding"))
    entry = {key: value for key, value in entry.items() if value is not None and value != ""}
    score = sum(a * b for a, b in zip(query_embedding, embedding))
    entry["score"] = round(score, 3)
    scored.append((score, entry))
  scored.sort(key=lambda item: item[0], reverse=True)
  return [entry for _, entry in scored[:limit]]


def log_anomaly(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  summary: str,
) -> int:
  """Save a human-readable anomaly tied to a benchmark run."""
  date = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    cursor = conn.execute(
      """
      INSERT INTO anomalies(agent_id, run_id, date, summary)
      VALUES (?, ?, ?, ?)
      """,
      (agent_id, run_id, date, summary),
    )
    anomaly_id = int(cursor.lastrowid)
    row = conn.execute("SELECT * FROM anomalies WHERE anomaly_id = ?", (anomaly_id,)).fetchone()
    _insert_main_notification(conn, agent_id=agent_id, run_id=run_id, kind="anomaly_logged", table_name="anomalies", row=row)
  return anomaly_id


def _insert_main_notification(conn: sqlite3.Connection, *, agent_id: int, run_id: int, kind: str, table_name: str, row: sqlite3.Row) -> None:
  """Queue one written SQLite row for the main agent to inspect."""
  notify_main(conn, agent_id=agent_id, run_id=run_id, kind=kind, table_name=table_name, row=dict(row))


def upsert_doc_chunk(
  conn: sqlite3.Connection,
  *,
  source: str,
  chunk_id: str,
  text: str,
  embedding: Sequence[float],
  title: str = "",
  url: str = "",
) -> int:
  """Insert or update one SGLang docs chunk and its embedding."""
  updated_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    row = conn.execute(
      """
      INSERT INTO doc_chunks(source, chunk_id, title, url, text, embedding, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(source, chunk_id) DO UPDATE SET
        title = excluded.title,
        url = excluded.url,
        text = excluded.text,
        embedding = excluded.embedding,
        updated_at = excluded.updated_at
      RETURNING doc_chunk_id
      """,
      (source, chunk_id, title, url, text, encode_embedding(embedding), updated_at),
    ).fetchone()
  return int(row["doc_chunk_id"])
