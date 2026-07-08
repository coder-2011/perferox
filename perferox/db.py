"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence


AGENT_RUNNING = "running"

RUN_STARTED = "started"
RUN_FINISHED = "finished"
RUN_FAILED = "failed"
RUN_STOPPED = "stopped"

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


@dataclass(frozen=True, slots=True)
class RunReservation:
  """Result from trying to reserve the next benchmark run id."""

  started: bool
  agent_id: int
  run_id: int | None
  reason: str
  successes: int
  attempts: int
  experiment_cap: int
  attempt_cap: int | None


def connect(path: str | Path) -> sqlite3.Connection:
  """Open one SQLite connection for a worker or tool call."""
  conn = sqlite3.connect(path, isolation_level=None)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA busy_timeout = 5000")
  conn.execute("PRAGMA journal_mode = WAL")
  return conn


def init_db(conn: sqlite3.Connection) -> None:
  """Create the Perferox tables and indexes if they do not already exist."""
  conn.executescript(
    """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS agents (
      agent_id INTEGER PRIMARY KEY CHECK(agent_id >= 0),
      kind TEXT NOT NULL,
      goal TEXT NOT NULL DEFAULT '',
      experiment_cap INTEGER NOT NULL CHECK(experiment_cap >= 0),
      attempt_cap INTEGER CHECK(attempt_cap IS NULL OR attempt_cap >= 0),
      status TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running', 'stopping', 'done', 'failed')),
      stop_requested INTEGER NOT NULL DEFAULT 0 CHECK(stop_requested IN (0, 1)),
      created_at TEXT NOT NULL,
      finished_at TEXT
    );

    CREATE TABLE IF NOT EXISTS runs (
      agent_id INTEGER NOT NULL,
      run_id INTEGER NOT NULL CHECK(run_id >= 0),
      gpu TEXT NOT NULL DEFAULT '',
      started_at TEXT NOT NULL,
      finished_at TEXT,
      status TEXT NOT NULL CHECK(status IN ('started', 'finished', 'failed', 'stopped')),
      trace_ref TEXT,
      command TEXT NOT NULL DEFAULT '',
      exact_hash TEXT NOT NULL UNIQUE,
      error TEXT NOT NULL DEFAULT '',
      PRIMARY KEY(agent_id, run_id),
      FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
    );

    CREATE TABLE IF NOT EXISTS experiments (
      agent_id INTEGER NOT NULL,
      run_id INTEGER NOT NULL,
      intent_key TEXT NOT NULL,
      intent_embedding TEXT,
      request_rps REAL,
      input_tps REAL,
      output_tps REAL,
      ttft_p50_ms REAL,
      ttft_p99_ms REAL,
      tpot_p50_ms REAL,
      tpot_p99_ms REAL,
      error_rate REAL,
      cache_hit_rate REAL,
      peak_gpu_mem_gb REAL,
      startup_s REAL,
      warmup_s REAL,
      accept_length REAL,
      correctness_score REAL,
      PRIMARY KEY(agent_id, run_id),
      FOREIGN KEY(agent_id, run_id) REFERENCES runs(agent_id, run_id)
    );

    CREATE TABLE IF NOT EXISTS anomalies (
      anomaly_id INTEGER PRIMARY KEY,
      agent_id INTEGER NOT NULL,
      run_id INTEGER NOT NULL,
      date TEXT NOT NULL,
      summary TEXT NOT NULL,
      FOREIGN KEY(agent_id, run_id) REFERENCES runs(agent_id, run_id)
    );

    CREATE TABLE IF NOT EXISTS explore_events (
      event_id INTEGER PRIMARY KEY,
      created_at TEXT NOT NULL,
      agent_id INTEGER,
      run_id INTEGER,
      source TEXT NOT NULL,
      kind TEXT NOT NULL,
      message TEXT NOT NULL,
      embedding TEXT,
      CHECK(run_id IS NULL OR agent_id IS NOT NULL),
      FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
      FOREIGN KEY(agent_id, run_id) REFERENCES runs(agent_id, run_id)
    );

    CREATE TABLE IF NOT EXISTS explore_summaries (
      summary_id INTEGER PRIMARY KEY,
      created_at TEXT NOT NULL,
      up_to_event_id INTEGER NOT NULL,
      summary TEXT NOT NULL,
      embedding TEXT,
      FOREIGN KEY(up_to_event_id) REFERENCES explore_events(event_id)
    );

    CREATE TABLE IF NOT EXISTS doc_chunks (
      doc_chunk_id INTEGER PRIMARY KEY,
      source TEXT NOT NULL,
      chunk_id TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',
      url TEXT NOT NULL DEFAULT '',
      text TEXT NOT NULL,
      embedding TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(source, chunk_id)
    );

    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    CREATE INDEX IF NOT EXISTS idx_runs_exact_hash ON runs(exact_hash);
    CREATE INDEX IF NOT EXISTS idx_experiments_intent_key ON experiments(intent_key);
    CREATE INDEX IF NOT EXISTS idx_anomalies_date ON anomalies(date);
    CREATE INDEX IF NOT EXISTS idx_explore_events_agent ON explore_events(agent_id, run_id);
    CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON doc_chunks(source);
    """
  )


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Iterator[None]:
  """Run a write transaction that takes SQLite's writer lock immediately."""
  conn.execute("BEGIN IMMEDIATE")
  try:
    yield
  except Exception:
    conn.rollback()
    raise
  conn.commit()


def utc_now() -> str:
  """Return a compact UTC timestamp for persisted rows."""
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode_embedding(embedding: Sequence[float] | None) -> str | None:
  """Store embeddings as deterministic JSON until a vector extension is needed."""
  if embedding is None:
    return None
  values = [float(value) for value in embedding]
  return json.dumps(values, separators=(",", ":"))


def create_agent(
  conn: sqlite3.Connection,
  *,
  kind: str,
  goal: str = "",
  experiment_cap: int,
  attempt_cap: int | None = None,
) -> int:
  """Allocate the next deterministic agent id and save its cap settings."""
  with write_tx(conn):
    row = conn.execute(
      "SELECT COALESCE(MAX(agent_id) + 1, 0) AS agent_id FROM agents"
    ).fetchone()
    agent_id = int(row["agent_id"])
    conn.execute(
      """
      INSERT INTO agents(agent_id, kind, goal, experiment_cap, attempt_cap, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      """,
      (agent_id, kind, goal, experiment_cap, attempt_cap, AGENT_RUNNING, utc_now()),
    )
  return agent_id


def request_stop(conn: sqlite3.Connection, agent_id: int | None = None) -> int:
  """Set stop_requested so future run reservations are refused."""
  with write_tx(conn):
    cursor = conn.execute(
      """
      UPDATE agents
      SET stop_requested = 1,
        status = CASE WHEN status = 'running' THEN 'stopping' ELSE status END
      WHERE (? IS NULL OR agent_id = ?) AND status IN ('running', 'stopping')
      """,
      (agent_id, agent_id),
    )
  return cursor.rowcount


def set_agent_status(conn: sqlite3.Connection, agent_id: int, status: str) -> None:
  """Update an agent lifecycle status."""
  finished_at = utc_now() if status in {"done", "failed"} else None
  stop_requested = 0 if status == AGENT_RUNNING else 1
  with write_tx(conn):
    cursor = conn.execute(
      """
      UPDATE agents
      SET status = ?, stop_requested = ?, finished_at = COALESCE(?, finished_at)
      WHERE agent_id = ?
      """,
      (status, stop_requested, finished_at, agent_id),
    )
  if cursor.rowcount != 1:
    raise ValueError(f"unknown agent_id: {agent_id}")


def reserve_run(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  exact_hash: str,
  gpu: str = "",
  command: str = "",
  trace_ref: str | None = None,
) -> RunReservation:
  """Reserve the next run id if stop state and caps allow a benchmark to start."""
  def blocked(reason: str) -> RunReservation:
    """Return a refusal result with the current cap counters."""
    return RunReservation(False, agent_id, None, reason, successes, attempts, experiment_cap, attempt_cap)

  with write_tx(conn):
    agent = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
    if agent is None:
      raise ValueError(f"unknown agent_id: {agent_id}")

    successes, attempts, in_flight = _run_counts(conn, agent_id)
    experiment_cap = int(agent["experiment_cap"])
    attempt_cap = agent["attempt_cap"]

    if agent["status"] != AGENT_RUNNING:
      return blocked(f"agent is {agent['status']}")
    if int(agent["stop_requested"]):
      return blocked("stop requested")
    if find_run_by_hash(conn, exact_hash) is not None:
      return blocked("exact repeat")
    if successes + in_flight >= experiment_cap:
      return blocked("success cap reached")
    if attempt_cap is not None and attempts >= int(attempt_cap):
      return blocked("attempt cap reached")

    # Count and insert under the same writer lock so two workers cannot share a run id.
    row = conn.execute(
      "SELECT COALESCE(MAX(run_id) + 1, 0) AS run_id FROM runs WHERE agent_id = ?",
      (agent_id,),
    ).fetchone()
    run_id = int(row["run_id"])
    conn.execute(
      """
      INSERT INTO runs(agent_id, run_id, gpu, started_at, status, trace_ref, command, exact_hash)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      (agent_id, run_id, gpu, utc_now(), RUN_STARTED, trace_ref, command, exact_hash),
    )
  return RunReservation(
    True, agent_id, run_id, "started", successes, attempts + 1, experiment_cap, attempt_cap
  )


def finish_run(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  status: str = RUN_FAILED,
  error: str = "",
  trace_ref: str | None = None,
) -> None:
  """Mark a started run as failed or stopped."""
  if status not in {RUN_FAILED, RUN_STOPPED}:
    raise ValueError(f"finish_run cannot mark successful runs: {status}")

  with write_tx(conn):
    cursor = conn.execute(
      """
      UPDATE runs
      SET status = ?, finished_at = ?, error = ?, trace_ref = COALESCE(?, trace_ref)
      WHERE agent_id = ? AND run_id = ? AND status = 'started'
      """,
      (status, utc_now(), error, trace_ref, agent_id, run_id),
    )
  if cursor.rowcount != 1:
    raise ValueError(f"run is not started: agent_id={agent_id} run_id={run_id}")


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  intent_key: str,
  intent_embedding: Sequence[float] | None = None,
  metrics: Mapping[str, float | int | None] | None = None,
) -> None:
  """Atomically save benchmark metrics and mark the run successful."""
  metric_values_by_name = metrics or {}
  unknown = sorted(set(metric_values_by_name) - set(METRIC_COLUMNS))
  if unknown:
    raise ValueError(f"unknown metric columns: {', '.join(unknown)}")

  columns = ", ".join(METRIC_COLUMNS)
  placeholders = ", ".join("?" for _ in METRIC_COLUMNS)
  metric_values = [metric_values_by_name.get(column) for column in METRIC_COLUMNS]

  with write_tx(conn):
    run = conn.execute(
      "SELECT status FROM runs WHERE agent_id = ? AND run_id = ?",
      (agent_id, run_id),
    ).fetchone()
    if run is None:
      raise ValueError(f"unknown run: agent_id={agent_id} run_id={run_id}")
    if run["status"] != RUN_STARTED:
      raise ValueError(f"run is not started: agent_id={agent_id} run_id={run_id}")

    conn.execute(
      f"""
      INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, {columns})
      VALUES (?, ?, ?, ?, {placeholders})
      """,
      (agent_id, run_id, intent_key, encode_embedding(intent_embedding), *metric_values),
    )
    conn.execute(
      "UPDATE runs SET status = ?, finished_at = ? WHERE agent_id = ? AND run_id = ?",
      (RUN_FINISHED, utc_now(), agent_id, run_id),
    )


def find_experiment_by_hash(conn: sqlite3.Connection, exact_hash: str) -> sqlite3.Row | None:
  """Return an exact experiment repeat if SQLite has already seen it."""
  return conn.execute(
    """
    SELECT experiments.*, runs.exact_hash, runs.gpu, runs.command, runs.finished_at
    FROM experiments
    JOIN runs USING(agent_id, run_id)
    WHERE runs.exact_hash = ?
    """,
    (exact_hash,),
  ).fetchone()


def find_run_by_hash(conn: sqlite3.Connection, exact_hash: str) -> sqlite3.Row | None:
  """Return any reserved or completed run with the exact experiment hash."""
  return conn.execute("SELECT * FROM runs WHERE exact_hash = ?", (exact_hash,)).fetchone()


def find_experiments_by_intent_key(
  conn: sqlite3.Connection,
  intent_key: str,
  limit: int = 20,
) -> list[sqlite3.Row]:
  """Return recent experiments with the same human-readable intent key."""
  return list(conn.execute(
    """
    SELECT experiments.*, runs.exact_hash, runs.gpu, runs.command, runs.finished_at
    FROM experiments
    JOIN runs USING(agent_id, run_id)
    WHERE experiments.intent_key = ?
    ORDER BY runs.finished_at DESC
    LIMIT ?
    """,
    (intent_key, limit),
  ))


def log_anomaly(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  summary: str,
) -> int:
  """Save a human-readable anomaly tied to a benchmark run."""
  cursor = conn.execute(
    """
    INSERT INTO anomalies(agent_id, run_id, date, summary)
    VALUES (?, ?, ?, ?)
    """,
    (agent_id, run_id, utc_now(), summary),
  )
  return int(cursor.lastrowid)


def log_explore_event(
  conn: sqlite3.Connection,
  *,
  source: str,
  kind: str,
  message: str,
  agent_id: int | None = None,
  run_id: int | None = None,
  embedding: Sequence[float] | None = None,
) -> int:
  """Append a raw ExploreState event without replacing history."""
  cursor = conn.execute(
    """
    INSERT INTO explore_events(created_at, agent_id, run_id, source, kind, message, embedding)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (utc_now(), agent_id, run_id, source, kind, message, encode_embedding(embedding)),
  )
  return int(cursor.lastrowid)


def log_explore_summary(
  conn: sqlite3.Connection,
  *,
  up_to_event_id: int,
  summary: str,
  embedding: Sequence[float] | None = None,
) -> int:
  """Append a compact ExploreState summary up to a raw event id."""
  cursor = conn.execute(
    """
    INSERT INTO explore_summaries(created_at, up_to_event_id, summary, embedding)
    VALUES (?, ?, ?, ?)
    """,
    (utc_now(), up_to_event_id, summary, encode_embedding(embedding)),
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
    (source, chunk_id, title, url, text, encode_embedding(embedding), utc_now()),
  ).fetchone()
  return int(row["doc_chunk_id"])


def _run_counts(conn: sqlite3.Connection, agent_id: int) -> tuple[int, int, int]:
  """Count successful and started benchmark runs for cap enforcement."""
  row = conn.execute(
    """
    SELECT
      SUM(CASE WHEN status = 'finished' THEN 1 ELSE 0 END) AS successes,
      SUM(CASE WHEN status = 'started' THEN 1 ELSE 0 END) AS in_flight,
      COUNT(*) AS attempts
    FROM runs
    WHERE agent_id = ?
    """,
    (agent_id,),
  ).fetchone()
  return int(row["successes"] or 0), int(row["attempts"] or 0), int(row["in_flight"] or 0)
