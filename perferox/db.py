"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Mapping, Sequence


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
  """Create the Perferox tables and indexes if they do not already exist."""
  schema_path = Path(__file__).with_name("init-db.sql")
  conn.executescript(schema_path.read_text(encoding="utf-8"))


def encode_embedding(embedding: Sequence[float]) -> str:
  """Store embeddings as deterministic JSON until a vector extension is needed."""
  values = [float(value) for value in embedding]
  return json.dumps(values, separators=(",", ":"))


@cache
def _embedder():
  """Load the Hugging Face embedding model once."""
  from sentence_transformers import SentenceTransformer
  return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_intent(intent_key: str) -> list[float]:
  """Embed a human-readable experiment intent key."""
  embedding = _embedder().encode(intent_key, normalize_embeddings=True)
  return [float(value) for value in embedding]


def create_agent(
  conn: sqlite3.Connection,
  *,
  kind: str,
  goal: str = "",
  experiment_cap: int,
  attempt_cap: int | None = None,
) -> int:
  """Allocate the next deterministic agent id and save its cap settings."""
  created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
  with conn:
    row = conn.execute(
      """
      INSERT INTO agents(agent_id, kind, goal, experiment_cap, attempt_cap, created_at)
      SELECT COALESCE(MAX(agent_id) + 1, 0), ?, ?, ?, ?, ? FROM agents
      RETURNING agent_id
      """,
      (kind, goal, experiment_cap, attempt_cap, created_at),
    ).fetchone()
  return int(row["agent_id"])


def request_stop(conn: sqlite3.Connection, agent_id: int | None = None) -> int:
  """Set stop_requested so future benchmark starts are refused."""
  with conn:
    cursor = conn.execute(
      """
      UPDATE agents
      SET stop_requested = 1
      WHERE (? IS NULL OR agent_id = ?) AND stop_requested = 0
      """,
      (agent_id, agent_id),
    )
  return cursor.rowcount


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  intent_key: str,
  metrics: Mapping[str, float | int | None] | None = None,
) -> None:
  """Atomically save benchmark metrics and mark the run successful."""
  metrics = metrics or {}
  unknown = sorted(set(metrics) - set(METRIC_COLUMNS))
  if unknown:
    raise ValueError(f"unknown metric columns: {', '.join(unknown)}")

  columns = ", ".join(METRIC_COLUMNS)
  placeholders = ", ".join("?" for _ in METRIC_COLUMNS)
  values = [metrics.get(column) for column in METRIC_COLUMNS]
  intent_embedding = embed_intent(intent_key)
  finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

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


def find_similar_experiments(
  conn: sqlite3.Connection,
  intent_key: str,
  limit: int = 20,
) -> list[tuple[float, sqlite3.Row]]:
  """Return experiments ranked by embedding similarity to an intent key."""
  query_embedding = embed_intent(intent_key)
  rows = conn.execute(
    """
    SELECT experiments.*, runs.exact_hash, runs.gpu, runs.command, runs.finished_at
    FROM experiments
    JOIN runs USING(agent_id, run_id)
    """
  ).fetchall()
  scored = [
    (sum(a * b for a, b in zip(query_embedding, json.loads(row["intent_embedding"]))), row)
    for row in rows
  ]
  scored.sort(key=lambda item: item[0], reverse=True)
  return scored[:limit]


def log_anomaly(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  summary: str,
) -> int:
  """Save a human-readable anomaly tied to a benchmark run."""
  date = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
  updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
