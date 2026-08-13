"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_EMBEDDER = None
_SCHEMA_VERSION = 3

METRIC_COLUMNS = ["request_rps", "input_tps", "output_tps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "error_rate", "cache_hit_rate", "peak_gpu_mem_gb", "startup_s", "warmup_s", "accept_length", "correctness_score"]
_METRIC_COLUMN_SET = set(METRIC_COLUMNS)
_METRIC_COLUMNS_SQL = ", ".join(METRIC_COLUMNS)
_METRIC_PLACEHOLDERS_SQL = ", ".join("?" for _ in METRIC_COLUMNS)
_METRIC_SELECT_SQL = ", ".join(f"e.{column}" for column in METRIC_COLUMNS)
def connect(path: str | Path, *, readonly: bool = False, immutable: bool = False) -> sqlite3.Connection:
  """Open one SQLite connection for a worker or tool call."""
  if immutable and not readonly:
    raise ValueError("immutable connections must be read-only")
  if readonly:
    immutable_param = "&immutable=1" if immutable else ""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + f"?mode=ro{immutable_param}", uri=True)
  else:
    conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA busy_timeout = 5000")
  mode_pragma = "PRAGMA query_only = ON" if readonly else f"PRAGMA journal_mode = {'WAL' if _wal_is_safe() else 'DELETE'}"
  conn.execute(mode_pragma)
  return conn


def _wal_is_safe() -> bool:
  """Return whether the linked SQLite contains the WAL-reset corruption fix."""
  version = sqlite3.sqlite_version_info
  backport = (3, 44, 6) <= version < (3, 45, 0) or (3, 50, 7) <= version < (3, 51, 0)
  return backport or version >= (3, 51, 3)


def init_db(conn: sqlite3.Connection) -> None:
  """Migrate the database to the current schema once per version."""
  # Dashboard polls call this every second, so skip idempotent DDL after migration.
  if conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
    return
  schema_path = Path(__file__).with_name("init-db.sql")
  schema = schema_path.read_text(encoding="utf-8")
  with conn:
    # Start inside executescript so its implicit commit cannot release the migration lock.
    conn.executescript(f"BEGIN IMMEDIATE;\n{schema}")
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
      return
    # Existing pre-beta databases need the new text fields added in place.
    for table, names in (
      ("runs", ("repository", "commit_hash", "provider", "server_command", "model_state")),
      ("agent_sessions", ("provider", "resource_id", "error", "llm_model", "reasoning_effort")),
    ):
      columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
      for name in names:
        if name not in columns:
          conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
    # Pack legacy JSON vectors once so every similarity search can use one matrix.
    for table, column in (("experiments", "intent_embedding"), ("doc_chunks", "embedding")):
      rows = conn.execute(f"SELECT rowid, {column} FROM {table} WHERE typeof({column}) = 'text' AND {column} != ''").fetchall()
      conn.executemany(
        f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
        ((_pack_embedding(json.loads(row[column])), row["rowid"]) for row in rows),
      )
      conn.execute(f"UPDATE {table} SET {column} = X'' WHERE typeof({column}) = 'text' AND {column} = ''")
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def embed_intent(intent_key: str) -> list[float]:
  """Encode one intent with the process-wide normalized embedding model."""
  return list(map(float, _embed_intents((intent_key,))[0]))


def _embed_intents(intent_keys: Sequence[str]):
  """Encode a batch through one process-wide normalized embedding model."""
  global _EMBEDDER
  if _EMBEDDER is None:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  return _EMBEDDER.encode(intent_keys, normalize_embeddings=True)


def _pack_embedding(embedding: object) -> bytes:
  """Pack one embedding as portable little-endian float32 bytes."""
  import numpy as np

  vector = np.asarray(embedding, dtype="<f4")
  if vector.ndim != 1:
    raise ValueError("embedding must be one-dimensional")
  return vector.tobytes()


def embedding_matrix(embeddings: Sequence[bytes]):
  """View equally sized packed embeddings as one float32 matrix."""
  import numpy as np

  if not embeddings:
    return np.empty((0, 0), dtype=np.float32)
  row_bytes = len(embeddings[0])
  if not row_bytes or row_bytes % 4 or any(len(embedding) != row_bytes for embedding in embeddings):
    raise ValueError("embeddings must be non-empty fixed-width float32 blobs")
  return np.frombuffer(b"".join(embeddings), dtype="<f4").reshape(len(embeddings), row_bytes // 4)


def read_explorer_state(conn: sqlite3.Connection, limit: int = 200) -> list[str]:
  """Return the latest compact ExplorerState lines in insertion order."""
  rows = conn.execute("SELECT line FROM explorer_state_lines ORDER BY line_id DESC LIMIT ?", (limit,)).fetchall()
  return [row["line"] for row in reversed(rows)]


def append_explorer_state(conn: sqlite3.Connection, *, agent_id: int | None, line: str) -> int:
  """Append one compact ExplorerState line."""
  created_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    row = conn.execute("INSERT INTO explorer_state_lines(agent_id, created_at, line) VALUES (?, ?, ?) RETURNING line_id", (agent_id, created_at, line)).fetchone()
  return int(row[0])


def record_agent_session(
  conn: sqlite3.Connection,
  *,
  session_name: str,
  role: str,
  agent_id: int | None = None,
  trace_ref: str = "",
  llm_model: str = "",
  reasoning_effort: str = "",
) -> None:
  """Record one tmux-wrapped process without clearing an accepted stop."""
  with conn:
    conn.execute(
      """
      INSERT INTO agent_sessions(session_name, role, agent_id, status, trace_ref, llm_model, reasoning_effort)
      VALUES (?, ?, ?, 'running', ?, ?, ?)
      ON CONFLICT(session_name) DO UPDATE SET
        role = excluded.role,
        agent_id = excluded.agent_id,
        status = CASE WHEN agent_sessions.status = 'ending' OR (agent_sessions.role = 'subagent' AND agent_sessions.status IN ('exited', 'failed')) THEN agent_sessions.status ELSE 'running' END,
        trace_ref = excluded.trace_ref,
        llm_model = excluded.llm_model,
        reasoning_effort = excluded.reasoning_effort,
        error = CASE WHEN agent_sessions.role = 'subagent' AND agent_sessions.status IN ('exited', 'failed') THEN agent_sessions.error ELSE '' END
      """,
      (session_name, role, agent_id, trace_ref, llm_model, reasoning_effort),
    )


def finish_agent_session(conn: sqlite3.Connection, *, session_name: str, status: str, trace_ref: str | None = None, error: str = "") -> bool:
  """Mark a tmux-wrapped agent process terminal with a bounded diagnosis."""
  if status not in {"exited", "failed", "missing"}:
    raise ValueError(f"invalid terminal agent status: {status}")
  with conn:
    return conn.execute(
      "UPDATE agent_sessions SET status = ?, error = ? WHERE session_name = ? AND status IN ('running', 'ending') AND (? IS NULL OR trace_ref = ?)",
      (status, error[:2000], session_name, trace_ref, trace_ref),
    ).rowcount == 1


def request_agent_end(conn: sqlite3.Connection, *, session_name: str) -> bool:
  """Mark a running agent session ending after a human End action."""
  with conn:
    return conn.execute("UPDATE agent_sessions SET status = 'ending' WHERE session_name = ? AND status = 'running'", (session_name,)).rowcount == 1


def request_soft_stop(conn: sqlite3.Connection) -> int:
  """Mark every running tmux-wrapped agent session as ending."""
  with conn:
    return conn.execute("UPDATE agent_sessions SET status = 'ending' WHERE status = 'running'").rowcount


def stop_requested(conn: sqlite3.Connection, *, agent_id: int) -> bool:
  """Return whether the main run or this subagent has been asked to stop."""
  return conn.execute(
    "SELECT 1 FROM agent_sessions WHERE status = 'ending' AND (session_name = 'perferox-main' OR (role = 'subagent' AND agent_id = ?)) LIMIT 1",
    (agent_id,),
  ).fetchone() is not None


def reserve_subagent(conn: sqlite3.Connection, *, active_cap: int, minimum_id: int = 0) -> int:
  """Atomically enforce stop/cap state and reserve the next agent id."""
  with conn:
    # The write lock keeps concurrent ToolNode delegations from choosing one id.
    conn.execute("BEGIN IMMEDIATE")
    state = conn.execute(
      """SELECT
        EXISTS(SELECT 1 FROM agent_sessions WHERE session_name = 'perferox-main' AND status = 'ending') AS stopped,
        (SELECT COUNT(*) FROM agent_sessions WHERE status IN ('running', 'ending') AND role = 'subagent') AS active,
        (SELECT COALESCE(MAX(agent_id) + 1, 0) FROM (SELECT agent_id FROM agent_sessions UNION ALL SELECT agent_id FROM runs)) AS agent_id
      """
    ).fetchone()
    if state["stopped"]:
      raise ValueError("stop requested; not starting a new benchmark subagent")
    if state["active"] >= active_cap:
      raise ValueError(f"max active subagents reached ({state['active']}/{active_cap})")
    agent_id = max(minimum_id, int(state["agent_id"]))
    conn.execute(
      "INSERT INTO agent_sessions(session_name, role, agent_id, status) VALUES (?, 'subagent', ?, 'running')",
      (f"perferox-agent-{agent_id}", agent_id),
    )
  return agent_id


def read_main_notifications(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
  """Return unread write notifications without acknowledging them."""
  return conn.execute(
    "SELECT * FROM main_notifications WHERE delivered_at IS NULL ORDER BY notification_id LIMIT ?",
    (limit,),
  ).fetchall()


def acknowledge_main_notifications(conn: sqlite3.Connection, notification_ids: list[int]) -> None:
  """Acknowledge notifications only after the coordinator processes them."""
  if not notification_ids:
    return
  delivered_at = datetime.now(UTC).isoformat(timespec="seconds")
  placeholders = ",".join("?" for _ in notification_ids)
  with conn:
    conn.execute(
      f"UPDATE main_notifications SET delivered_at = ? WHERE delivered_at IS NULL AND notification_id IN ({placeholders})",
      (delivered_at, *notification_ids),
    )


def notify_main(
  conn: sqlite3.Connection,
  *,
  agent_id: int | None,
  run_id: int | None,
  kind: str,
  table_name: str,
  row: Mapping[str, object] | sqlite3.Row,
) -> None:
  """Queue one host event for the main agent to inspect."""
  conn.execute(
    "INSERT INTO main_notifications(created_at, agent_id, run_id, kind, table_name, row_json) VALUES (?, ?, ?, ?, ?, ?)",
    (
      datetime.now(UTC).isoformat(timespec="seconds"), agent_id, run_id, kind, table_name,
      json.dumps(dict(row), separators=(",", ":"), default=str),
    ),
  )


def record_cloud_resource(conn: sqlite3.Connection, *, agent_id: int, provider: str, resource_id: str) -> None:
  """Attach one paid resource to its owning worker session."""
  with conn:
    updated = conn.execute(
      """UPDATE agent_sessions
         SET provider = ?, resource_id = ?
         WHERE role = 'subagent' AND agent_id = ? AND resource_id = ''""",
      (provider, resource_id, agent_id),
    ).rowcount
  if updated != 1:
    raise ValueError(f"agent {agent_id} has no available worker session")


def pending_cloud_resource(conn: sqlite3.Connection, *, agent_id: int) -> sqlite3.Row | None:
  """Return the worker-owned resource when it still needs termination."""
  return conn.execute(
    "SELECT * FROM agent_sessions WHERE role = 'subagent' AND agent_id = ? AND resource_id != ''",
    (agent_id,),
  ).fetchone()


def clear_cloud_resource(conn: sqlite3.Connection, *, agent_id: int) -> None:
  """Clear the worker-owned resource after successful teardown."""
  with conn:
    conn.execute(
      "UPDATE agent_sessions SET resource_id = '' WHERE role = 'subagent' AND agent_id = ?",
      (agent_id,),
    )


def start_benchmark_run(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  command: str,
  repository: str = "",
  commit: str = "",
  provider: str = "",
  gpu: str = "",
  server_command: str = "",
  model_state: str = "",
  trace_ref: str = "",
  attempt_cap: int | None = None,
) -> int:
  """Assign the next run id and insert the started benchmark row."""
  identity = (repository, commit, provider, gpu, server_command, model_state, command)
  exact_hash = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
  started_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    # Serialize the stop/cap checks and run-id assignment with the insert.
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
      """
      INSERT INTO runs(agent_id, run_id, repository, commit_hash, provider, gpu, server_command, model_state, started_at, trace_ref, command, exact_hash)
      SELECT ?, COALESCE(MAX(run_id) + 1, 0), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
      FROM runs WHERE agent_id = ?
      RETURNING *
      """,
      (agent_id, repository, commit, provider, gpu, server_command, model_state, started_at, trace_ref, command, exact_hash, agent_id),
    ).fetchone()
    run_id = int(row["run_id"])
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_started", table_name="runs", row=row)
  return run_id


def mark_run_failed(conn: sqlite3.Connection, *, agent_id: int, run_id: int, error: str) -> None:
  """Mark a started benchmark run as finished with an error."""
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    row = conn.execute(
      "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL RETURNING *",
      (finished_at, error[:2000], agent_id, run_id),
    ).fetchone()
    if row is None:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_failed", table_name="runs", row=row)


def mark_run_succeeded(conn: sqlite3.Connection, *, agent_id: int, run_id: int, metrics: Mapping[str, float]) -> None:
  """Finish one successful run with canonical host-parsed metrics."""
  unknown = sorted(set(metrics) - _METRIC_COLUMN_SET)
  if unknown:
    raise ValueError(f"unknown host metric columns: {', '.join(unknown)}")
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  values = [metrics.get(column) for column in METRIC_COLUMNS]
  with conn:
    row = conn.execute(
      "UPDATE runs SET finished_at = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL RETURNING *",
      (finished_at, agent_id, run_id),
    ).fetchone()
    if row is None:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    experiment = conn.execute(
      f"INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, {_METRIC_COLUMNS_SQL}) VALUES (?, ?, '', X'', {_METRIC_PLACEHOLDERS_SQL}) RETURNING *",
      (agent_id, run_id, *values),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_succeeded", table_name="experiments", row=experiment)


def fail_unfinished_runs(conn: sqlite3.Connection, *, agent_id: int, error: str) -> int:
  """Close every run abandoned by a terminal worker process."""
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    rows = conn.execute(
      "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND finished_at IS NULL RETURNING *",
      (finished_at, error[:2000], agent_id),
    ).fetchall()
    for row in rows:
      notify_main(conn, agent_id=agent_id, run_id=row["run_id"], kind="run_failed", table_name="runs", row=row)
  return len(rows)


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  intent_key: str,
) -> int:
  """Save an intent for the coordinator's next embedding batch."""
  with conn:
    row = conn.execute(
      "UPDATE experiments SET intent_key = ? WHERE agent_id = ? AND run_id = ? AND intent_key = '' RETURNING *",
      (intent_key, agent_id, run_id),
    ).fetchone()
    if row is None:
      raise ValueError(f"unknown, unsuccessful, or annotated run: agent_id={agent_id} run_id={run_id}")
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="experiment_logged", table_name="experiments", row=row)
  return run_id


def embed_pending_intents(conn: sqlite3.Connection) -> int:
  """Batch and persist every intent left pending by benchmark workers."""
  rows = conn.execute(
    "SELECT agent_id, run_id, intent_key FROM experiments WHERE intent_key != '' AND length(intent_embedding) = 0 ORDER BY agent_id, run_id"
  ).fetchall()
  if not rows:
    return 0
  vectors = _embed_intents(tuple(row["intent_key"] for row in rows))
  if len(vectors) != len(rows):
    raise ValueError("embedding model returned the wrong batch size")
  with conn:
    conn.executemany(
      "UPDATE experiments SET intent_embedding = ? WHERE agent_id = ? AND run_id = ? AND intent_key = ? AND length(intent_embedding) = 0",
      ((_pack_embedding(vector), row["agent_id"], row["run_id"], row["intent_key"]) for row, vector in zip(rows, vectors)),
    )
  return len(rows)


def find_similar_experiments(conn: sqlite3.Connection, intent: str, limit: int = 5) -> list[dict[str, object]]:
  """Return logged experiments closest to an intent embedding."""
  rows = conn.execute(
    f"""
    SELECT e.agent_id, e.run_id, e.intent_key, e.intent_embedding, {_METRIC_SELECT_SQL},
      r.trace_ref, r.command, r.started_at, r.finished_at, r.error
    FROM experiments e
    JOIN runs r ON r.agent_id = e.agent_id AND r.run_id = e.run_id
    WHERE length(e.intent_embedding) > 0
    """
  ).fetchall()
  if not rows:
    return []
  import numpy as np

  vectors = embedding_matrix(tuple(row["intent_embedding"] for row in rows))
  query_embedding = np.asarray(embed_intent(intent), dtype=np.float32)
  if vectors.shape[1] != query_embedding.size:
    raise ValueError("stored and query embedding widths differ")
  scores = vectors @ query_embedding
  indices = np.argsort(-scores, kind="stable")[:limit]
  matches = []
  for index in indices:
    entry = dict(rows[index])
    entry.pop("intent_embedding")
    entry = {key: value for key, value in entry.items() if value is not None and value != ""}
    score = float(scores[index])
    entry["score"] = round(score, 3)
    matches.append(entry)
  return matches


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
    row = conn.execute(
      "INSERT INTO anomalies(agent_id, run_id, date, summary) VALUES (?, ?, ?, ?) RETURNING *",
      (agent_id, run_id, date, summary),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="anomaly_logged", table_name="anomalies", row=row)
  return int(row["anomaly_id"])
