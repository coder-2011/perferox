"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

_EMBEDDER = None

MetricName = Literal["request_rps", "input_tps", "output_tps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "error_rate", "cache_hit_rate", "peak_gpu_mem_gb", "startup_s", "warmup_s", "accept_length", "correctness_score"]
METRIC_COLUMNS = get_args(MetricName)
_METRIC_COLUMN_SET = set(METRIC_COLUMNS)
_METRIC_COLUMNS_SQL = ", ".join(METRIC_COLUMNS)
_METRIC_PLACEHOLDERS_SQL = ", ".join("?" for _ in METRIC_COLUMNS)
_METRIC_SELECT_SQL = ", ".join(f"e.{column}" for column in METRIC_COLUMNS)
_METRIC_SCHEMA_SQL = ", ".join(f"{column} REAL" for column in METRIC_COLUMNS)
_RATE_COLUMNS = {"error_rate", "cache_hit_rate"}
INTENT_SIMILARITY_THRESHOLD = 0.90
MAX_NOTIFICATION_VALUE_CHARS = 1000


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
  """Open one SQLite connection for a worker or tool call."""
  if readonly:
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
  else:
    conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA busy_timeout = 5000")
  mode_pragma = "PRAGMA query_only = ON" if readonly else "PRAGMA journal_mode = WAL"
  conn.execute(mode_pragma)
  return conn


@contextmanager
def open_db(path: str | Path, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
  """Open and always close one configured SQLite connection."""
  conn = connect(path, readonly=readonly)
  try:
    yield conn
  finally:
    conn.close()


def init_db(conn: sqlite3.Connection) -> None:
  """Create every table and index declared by the schema."""
  schema_path = Path(__file__).with_name("init-db.sql")
  conn.executescript(schema_path.read_text(encoding="utf-8"))
  _migrate_schema(conn)


def _migrate_schema(conn: sqlite3.Connection) -> None:
  """Add current run and notification fields to existing databases."""
  run_columns = ("environment", "spec_hash", "intent_key", "intent_embedding")
  existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
  for name in run_columns:
    if name not in existing:
      conn.execute(f"ALTER TABLE runs ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
  legacy_environment = {"repository", "target_commit", "provider", "resource_config", "hardware_config", "server_config"}
  if legacy_environment <= existing:
    # Preserve the previous split representation as one stable JSON value.
    conn.execute(
      """
      UPDATE runs SET environment = json_object(
        'hardware_config', hardware_config, 'provider', provider,
        'repository', repository, 'resource_config', resource_config,
        'server_config', server_config, 'target_commit', target_commit
      ) WHERE environment = ''
      """
    )
  # Historical rows retain their command hash and experiment intent as the best available identity.
  conn.execute("UPDATE runs SET spec_hash = exact_hash WHERE spec_hash = ''")
  experiment_columns = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
  if {"intent_key", "intent_embedding"} <= experiment_columns:
    conn.execute(
      """
      UPDATE runs SET
        intent_key = COALESCE((SELECT intent_key FROM experiments e WHERE e.agent_id = runs.agent_id AND e.run_id = runs.run_id), ''),
        intent_embedding = COALESCE((SELECT intent_embedding FROM experiments e WHERE e.agent_id = runs.agent_id AND e.run_id = runs.run_id), '')
      WHERE intent_key = ''
      """
    )
    # Rebuild once so future writes have one canonical owner for intent data.
    conn.executescript(
      f"""
      ALTER TABLE experiments RENAME TO experiments_legacy;
      CREATE TABLE experiments (
        agent_id INTEGER NOT NULL, run_id INTEGER NOT NULL,
        {_METRIC_SCHEMA_SQL},
        PRIMARY KEY(agent_id, run_id),
        FOREIGN KEY(agent_id, run_id) REFERENCES runs(agent_id, run_id)
      );
      INSERT INTO experiments(agent_id, run_id, {_METRIC_COLUMNS_SQL})
      SELECT agent_id, run_id, {_METRIC_COLUMNS_SQL} FROM experiments_legacy;
      DROP TABLE experiments_legacy;
      """
    )
  resource_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cloud_resources)")}
  if "environment" not in resource_columns:
    conn.execute("ALTER TABLE cloud_resources ADD COLUMN environment TEXT NOT NULL DEFAULT '{}'")
  conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_spec_hash ON runs(spec_hash)")
  conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_intent_key ON runs(intent_key)")
  conn.commit()


def embed_intent(intent_key: str) -> list[float]:
  """Encode one intent with the process-wide normalized embedding model."""
  global _EMBEDDER
  if _EMBEDDER is None:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  return list(map(float, _EMBEDDER.encode(intent_key, normalize_embeddings=True)))


def read_explorer_state(conn: sqlite3.Connection, limit: int | None = None) -> list[str]:
  """Return all or a bounded recent ExplorerState window in insertion order."""
  if limit is None:
    return [row["line"] for row in conn.execute("SELECT line FROM explorer_state_lines ORDER BY line_id")]
  rows = conn.execute("SELECT line FROM explorer_state_lines ORDER BY line_id DESC LIMIT ?", (limit,)).fetchall()
  return [row["line"] for row in reversed(rows)]


def append_explorer_state(conn: sqlite3.Connection, *, agent_id: int | None, line: str) -> int:
  """Append one compact ExplorerState line."""
  created_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    row = conn.execute(
      """
      INSERT INTO explorer_state_lines(agent_id, created_at, line)
      VALUES (?, ?, ?)
      RETURNING line_id
      """,
      (agent_id, created_at, line),
    ).fetchone()
  return int(row["line_id"])


def record_agent_session(conn: sqlite3.Connection, *, session_name: str, role: str, agent_id: int | None = None, trace_ref: str = "") -> None:
  """Record one tmux-wrapped process without clearing an accepted stop."""
  with conn:
    conn.execute(
      """
      INSERT INTO agent_sessions(session_name, role, agent_id, status, trace_ref)
      VALUES (?, ?, ?, 'running', ?)
      ON CONFLICT(session_name) DO UPDATE SET
        role = excluded.role,
        agent_id = excluded.agent_id,
        status = CASE WHEN agent_sessions.status = 'ending' OR (agent_sessions.role = 'subagent' AND agent_sessions.status IN ('exited', 'failed')) THEN agent_sessions.status ELSE 'running' END,
        trace_ref = excluded.trace_ref
      """,
      (session_name, role, agent_id, trace_ref),
    )


def finish_agent_session(conn: sqlite3.Connection, *, session_name: str, status: str) -> bool:
  """Mark a tmux-wrapped agent process as exited or missing."""
  with conn:
    row = conn.execute(
      """
      UPDATE agent_sessions
      SET status = ?
      WHERE session_name = ? AND status IN ('starting', 'running', 'ending')
      RETURNING agent_id
      """,
      (status, session_name),
    ).fetchone()
    if row is not None and row["agent_id"] is not None and status in {"failed", "missing"}:
      _fail_unfinished_runs(conn, int(row["agent_id"]), f"worker session {status}: {session_name}")
  return row is not None


def request_agent_end(conn: sqlite3.Connection, *, session_name: str) -> bool:
  """Mark an active agent session ending after a human End action."""
  with conn:
    return conn.execute(
      "UPDATE agent_sessions SET status = 'ending' WHERE session_name = ? AND status IN ('starting', 'running')",
      (session_name,),
    ).rowcount == 1


def request_soft_stop(conn: sqlite3.Connection) -> int:
  """Mark every active tmux-wrapped agent session as ending."""
  with conn:
    return conn.execute("UPDATE agent_sessions SET status = 'ending' WHERE status IN ('starting', 'running')").rowcount


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


def reserve_subagent(conn: sqlite3.Connection, *, active_cap: int) -> int:
  """Atomically enforce stop/cap state and reserve the next agent id."""
  with conn:
    # The write lock keeps concurrent ToolNode delegations from choosing one id.
    conn.execute("BEGIN IMMEDIATE")
    main = conn.execute("SELECT status FROM agent_sessions WHERE session_name = 'perferox-main'").fetchone()
    if main is not None and main["status"] == "ending":
      raise ValueError("stop requested; not starting a new benchmark subagent")
    active = conn.execute("SELECT COUNT(*) FROM agent_sessions WHERE status IN ('starting', 'running', 'ending') AND role = 'subagent'").fetchone()[0]
    if active >= active_cap:
      raise ValueError(f"max active subagents reached ({active}/{active_cap})")
    row = conn.execute(
      """
      INSERT INTO agent_sessions(session_name, role, agent_id, status)
      SELECT 'perferox-agent-' || agent_id, 'subagent', agent_id, 'starting'
      FROM (
        SELECT COALESCE(MAX(agent_id) + 1, 0) AS agent_id FROM (
          SELECT agent_id FROM agent_sessions WHERE agent_id IS NOT NULL
          UNION ALL
          SELECT agent_id FROM runs
        )
      )
      RETURNING agent_id
      """
    ).fetchone()
  return int(row["agent_id"])


def activate_subagent(conn: sqlite3.Connection, *, agent_id: int, trace_ref: str) -> None:
  """Attach the trace and activate a successfully launched reservation."""
  with conn:
    conn.execute(
      """
      UPDATE agent_sessions
      SET trace_ref = ?, status = CASE WHEN status = 'starting' THEN 'running' ELSE status END
      WHERE role = 'subagent' AND agent_id = ?
      """,
      (trace_ref, agent_id),
    )


def read_main_notifications(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
  """Read unread notifications for the singleton main process."""
  if limit < 1:
    return []
  return conn.execute(
    "SELECT * FROM main_notifications WHERE delivered_at IS NULL ORDER BY notification_id LIMIT ?",
    (limit,),
  ).fetchall()


def ack_main_notifications(conn: sqlite3.Connection, notification_ids: list[int]) -> int:
  """Acknowledge successfully processed notification rows."""
  if not notification_ids:
    return 0
  placeholders = ",".join("?" for _ in notification_ids)
  delivered_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    cursor = conn.execute(
      f"UPDATE main_notifications SET delivered_at = ? WHERE delivered_at IS NULL AND notification_id IN ({placeholders})",
      (delivered_at, *notification_ids),
    )
  return cursor.rowcount


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
  compact_row = _compact_notification_row(row)
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
      json.dumps(compact_row, separators=(",", ":"), default=str),
    ),
  )


def _compact_notification_row(row: Mapping[str, object] | sqlite3.Row) -> dict[str, object]:
  """Remove vectors and bound large values in one notification payload."""
  compact = {}
  for key, value in dict(row).items():
    if key in {"embedding", "intent_embedding"}:
      continue
    if isinstance(value, str) and len(value) > MAX_NOTIFICATION_VALUE_CHARS:
      value = value[:MAX_NOTIFICATION_VALUE_CHARS] + "..."
    compact[key] = value
  return compact


def start_benchmark_run(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  environment: Mapping[str, object] | str,
  command: str,
  intent_key: str,
  trace_ref: str = "",
  attempt_cap: int | None = None,
) -> int:
  """Reserve a non-duplicate benchmark intent and full experiment spec."""
  command = command.strip()
  intent_key = " ".join(intent_key.split())
  if not command or not intent_key:
    raise ValueError("command and intent_key must not be empty")
  environment_json = _canonical_config("environment", environment)
  spec = {"command": command, "environment": environment_json}
  spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
  spec_hash = hashlib.sha256(spec_json.encode()).hexdigest()
  intent_embedding = embed_intent(intent_key)
  intent_json = json.dumps(intent_embedding, separators=(",", ":"))
  started_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    # Serialize stop, cap, repeat checks, run-id assignment, and reservation.
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
    duplicate = conn.execute(
      """
      SELECT r.agent_id, r.run_id FROM runs r
      WHERE r.spec_hash = ? AND r.error = ''
        AND (r.finished_at IS NULL OR EXISTS (
          SELECT 1 FROM experiments e WHERE e.agent_id = r.agent_id AND e.run_id = r.run_id
        ))
      LIMIT 1
      """,
      (spec_hash,),
    ).fetchone()
    if duplicate is not None:
      raise ValueError(f"exact experiment already active or successful: agent_id={duplicate['agent_id']} run_id={duplicate['run_id']}")
    similar = _find_blocking_intent(
      conn,
      intent_embedding,
      environment=environment_json,
    )
    if similar is not None:
      raise ValueError(
        f"similar intent already active or successful: agent_id={similar['agent_id']} "
        f"run_id={similar['run_id']} score={similar['score']:.3f} intent={similar['intent_key']}"
      )
    run_id = int(conn.execute("SELECT COALESCE(MAX(run_id) + 1, 0) FROM runs WHERE agent_id = ?", (agent_id,)).fetchone()[0])
    # Old databases may still enforce UNIQUE(exact_hash), so keep its compatibility value attempt-specific.
    exact_hash = hashlib.sha256(f"{spec_hash}:{agent_id}:{run_id}".encode()).hexdigest()
    row = conn.execute(
      """
      INSERT INTO runs(
        agent_id, run_id, started_at, trace_ref, command, exact_hash,
        environment, spec_hash, intent_key, intent_embedding
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      RETURNING *
      """,
      (
        agent_id, run_id, started_at, trace_ref, command, exact_hash,
        environment_json, spec_hash, intent_key, intent_json,
      ),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_started", table_name="runs", row=row)
  return run_id


def _canonical_config(name: str, value: Mapping[str, object] | str) -> str:
  """Return one stable non-empty configuration representation."""
  if isinstance(value, str):
    text = value.strip()
    if not text:
      raise ValueError(f"{name} must not be empty")
    try:
      parsed = json.loads(text)
    except json.JSONDecodeError:
      return text
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
  if not value:
    raise ValueError(f"{name} must not be empty")
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _find_blocking_intent(
  conn: sqlite3.Connection,
  query_embedding: list[float],
  *,
  environment: str,
) -> dict[str, object] | None:
  """Return a similar active or successful intent in the same target environment."""
  rows = conn.execute(
    """
    SELECT r.agent_id, r.run_id, r.intent_key, r.intent_embedding FROM runs r
    WHERE r.intent_embedding != '' AND r.error = ''
      AND r.environment = ?
      AND (r.finished_at IS NULL OR EXISTS (
        SELECT 1 FROM experiments e WHERE e.agent_id = r.agent_id AND e.run_id = r.run_id
      ))
    """,
    (environment,),
  ).fetchall()
  best = None
  for row in rows:
    embedding = json.loads(row["intent_embedding"])
    if len(embedding) != len(query_embedding):
      continue
    score = sum(a * b for a, b in zip(query_embedding, embedding))
    if score >= INTENT_SIMILARITY_THRESHOLD and (best is None or score > best["score"]):
      best = {"agent_id": row["agent_id"], "run_id": row["run_id"], "intent_key": row["intent_key"], "score": score}
  return best


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


def fail_unfinished_runs(conn: sqlite3.Connection, agent_id: int, error: str) -> int:
  """Fail every unfinished run left behind by one worker."""
  with conn:
    return _fail_unfinished_runs(conn, agent_id, error)


def _fail_unfinished_runs(conn: sqlite3.Connection, agent_id: int, error: str) -> int:
  """Fail unfinished rows inside the caller's transaction."""
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")
  rows = conn.execute(
    "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND finished_at IS NULL RETURNING *",
    (finished_at, error[:2000], agent_id),
  ).fetchall()
  for row in rows:
    notify_main(conn, agent_id=agent_id, run_id=int(row["run_id"]), kind="run_failed", table_name="runs", row=row)
  return len(rows)


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  run_id: int,
  metrics: Mapping[str, float | int | None] | None = None,
) -> int:
  """Atomically save benchmark metrics and mark the run successful."""
  metrics = metrics or {}
  if not metrics:
    raise ValueError("metrics must not be empty")
  unknown = sorted(set(metrics) - _METRIC_COLUMN_SET)
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
    if name in _RATE_COLUMNS and normalized > 1.0:
      normalized /= 100.0
    if not math.isfinite(normalized) or normalized < 0.0:
      raise ValueError(f"{name} must be finite and >= 0")
    if name in _RATE_COLUMNS and normalized > 1.0:
      raise ValueError(f"{name} must normalize to a 0..1 rate")
    normalized_metrics[name] = normalized
  if not normalized_metrics:
    raise ValueError("metrics must contain at least one numeric value")

  values = [normalized_metrics.get(column) for column in METRIC_COLUMNS]
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")

  with conn:
    # Claim the unfinished run and persist its experiment as one transaction.
    row = conn.execute(
      "UPDATE runs SET finished_at = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL AND error = '' RETURNING *",
      (finished_at, agent_id, run_id),
    ).fetchone()
    if row is None:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_succeeded", table_name="runs", row=row)
    row = conn.execute(
      f"""
      INSERT INTO experiments(agent_id, run_id, {_METRIC_COLUMNS_SQL})
      VALUES (?, ?, {_METRIC_PLACEHOLDERS_SQL})
      RETURNING *
      """,
      (agent_id, run_id, *values),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="experiment_logged", table_name="experiments", row=row)
  return run_id


def find_similar_experiments(conn: sqlite3.Connection, intent: str, limit: int = 5) -> list[dict[str, object]]:
  """Return logged experiments closest to an intent embedding."""
  query_embedding = embed_intent(intent)
  rows = conn.execute(
    f"""
    SELECT e.agent_id, e.run_id, r.intent_key, r.intent_embedding, {_METRIC_SELECT_SQL},
      r.environment, r.trace_ref, r.command, r.started_at, r.finished_at, r.error
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
    row = conn.execute(
      "INSERT INTO anomalies(agent_id, run_id, date, summary) VALUES (?, ?, ?, ?) RETURNING *",
      (agent_id, run_id, date, summary),
    ).fetchone()
    anomaly_id = int(row["anomaly_id"])
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="anomaly_logged", table_name="anomalies", row=row)
  return anomaly_id


def register_cloud_resource(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  provider: str,
  resource_id: str,
  environment: Mapping[str, object] | str,
) -> None:
  """Persist the one paid resource owned by a worker before SSH setup."""
  environment_json = _canonical_config("environment", environment)
  created_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    active = conn.execute(
      "SELECT resource_id FROM cloud_resources WHERE agent_id = ? AND terminated_at IS NULL",
      (agent_id,),
    ).fetchone()
    if active is not None:
      raise ValueError(f"agent already owns active resource {active['resource_id']}")
    conn.execute(
      """
      INSERT INTO cloud_resources(agent_id, provider, resource_id, environment, created_at)
      VALUES (?, ?, ?, ?, ?)
      """,
      (agent_id, provider, resource_id, environment_json, created_at),
    )


def active_cloud_resources(conn: sqlite3.Connection, *, agent_id: int) -> list[sqlite3.Row]:
  """Return resources that still require deterministic provider cleanup."""
  return conn.execute(
    "SELECT * FROM cloud_resources WHERE agent_id = ? AND terminated_at IS NULL ORDER BY created_at",
    (agent_id,),
  ).fetchall()


def finish_cloud_resource(
  conn: sqlite3.Connection,
  *,
  provider: str,
  resource_id: str,
  error: str = "",
) -> None:
  """Record successful cleanup or retain a bounded cleanup failure for retry."""
  terminated_at = None if error else datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    conn.execute(
      """
      UPDATE cloud_resources SET terminated_at = ?, cleanup_error = ?
      WHERE provider = ? AND resource_id = ? AND terminated_at IS NULL
      """,
      (terminated_at, error[:2000], provider, resource_id),
    )
