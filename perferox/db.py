"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import hashlib
import json
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


def connect(path: str | Path) -> sqlite3.Connection:
  """Open one SQLite connection for a worker or tool call."""
  conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA busy_timeout = 5000")
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
  return run_id


def mark_run_failed(conn: sqlite3.Connection, *, agent_id: int, run_id: int, error: str) -> None:
  """Mark a started benchmark run as finished with an error."""
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    conn.execute(
      "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND run_id = ?",
      (finished_at, error[:2000], agent_id, run_id),
    )


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
  values = [metrics.get(column) for column in METRIC_COLUMNS]
  intent_embedding = embed_intent(intent_key)
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")

  with conn:
    cursor = conn.execute(
      "UPDATE runs SET finished_at = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL",
      (finished_at, agent_id, run_id),
    )
    if cursor.rowcount != 1:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    conn.execute(
      f"""
      INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, {columns})
      VALUES (?, ?, ?, ?, {placeholders})
      """,
      (agent_id, run_id, intent_key, encode_embedding(intent_embedding), *values),
    )
  return run_id


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
  return int(cursor.lastrowid)


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
