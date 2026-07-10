"""SQLite schema and host-owned state transitions for Perferox."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

_EMBEDDER = None

METRIC_COLUMNS = ["request_rps", "input_tps", "output_tps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "error_rate", "cache_hit_rate", "peak_gpu_mem_gb", "startup_s", "warmup_s", "accept_length", "correctness_score"]
_METRIC_COLUMN_SET = set(METRIC_COLUMNS)
_METRIC_COLUMNS_SQL = ", ".join(METRIC_COLUMNS)
_METRIC_PLACEHOLDERS_SQL = ", ".join("?" for _ in METRIC_COLUMNS)
_METRIC_SELECT_SQL = ", ".join(f"e.{column}" for column in METRIC_COLUMNS)
_RATE_COLUMNS = {"error_rate", "cache_hit_rate"}


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


def init_db(conn: sqlite3.Connection) -> None:
  """Create every table and index declared by the schema."""
  schema_path = Path(__file__).with_name("init-db.sql")
  conn.executescript(schema_path.read_text(encoding="utf-8"))
  columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
  # Existing pre-beta databases need the identity fields added in place.
  for name in ("repository", "commit_hash", "provider", "server_command", "model_state"):
    if name not in columns:
      conn.execute(f"ALTER TABLE runs ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")


def embed_intent(intent_key: str) -> list[float]:
  """Encode one intent with the process-wide normalized embedding model."""
  global _EMBEDDER
  if _EMBEDDER is None:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  return list(map(float, _EMBEDDER.encode(intent_key, normalize_embeddings=True)))


def read_explorer_state(conn: sqlite3.Connection) -> list[str]:
  """Return compact ExplorerState lines in insertion order."""
  return [row["line"] for row in conn.execute("SELECT line FROM explorer_state_lines ORDER BY line_id")]


def append_explorer_state(conn: sqlite3.Connection, *, agent_id: int | None, line: str) -> int:
  """Append one compact ExplorerState line."""
  created_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    row = conn.execute("INSERT INTO explorer_state_lines(agent_id, created_at, line) VALUES (?, ?, ?) RETURNING line_id", (agent_id, created_at, line)).fetchone()
  return int(row[0])


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


def finish_agent_session(conn: sqlite3.Connection, *, session_name: str, status: str, trace_ref: str | None = None) -> bool:
  """Mark a tmux-wrapped agent process as exited or missing."""
  with conn:
    return conn.execute("UPDATE agent_sessions SET status = ? WHERE session_name = ? AND status IN ('running', 'ending') AND (? IS NULL OR trace_ref = ?)", (status, session_name, trace_ref, trace_ref)).rowcount == 1


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


def take_main_notifications(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
  """Return unread write notifications and mark them delivered."""
  delivered_at = datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    rows = conn.execute(
      "UPDATE main_notifications SET delivered_at = ? WHERE notification_id IN "
      "(SELECT notification_id FROM main_notifications WHERE delivered_at IS NULL ORDER BY notification_id LIMIT ?) RETURNING *",
      (delivered_at, limit),
    ).fetchall()
  return sorted(rows, key=lambda row: row["notification_id"])


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
  """Persist one paid resource before the worker can continue."""
  with conn:
    conn.execute(
      "INSERT INTO cloud_resources(provider, resource_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
      (provider, resource_id, agent_id, datetime.now(UTC).isoformat(timespec="seconds")),
    )


def pending_cloud_resources(conn: sqlite3.Connection, *, agent_id: int) -> list[sqlite3.Row]:
  """Return paid resources that still require termination."""
  return conn.execute(
    "SELECT * FROM cloud_resources WHERE agent_id = ? AND terminated_at IS NULL ORDER BY created_at",
    (agent_id,),
  ).fetchall()


def finish_cloud_resource(conn: sqlite3.Connection, *, provider: str, resource_id: str, error: str = "") -> None:
  """Record a provider teardown result for one paid resource."""
  terminated_at = None if error else datetime.now(UTC).isoformat(timespec="seconds")
  with conn:
    conn.execute(
      "UPDATE cloud_resources SET terminated_at = ?, termination_error = ? WHERE provider = ? AND resource_id = ?",
      (terminated_at, error[:2000], provider, resource_id),
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
  identity = {
    "repository": repository,
    "commit": commit,
    "provider": provider,
    "gpu": gpu,
    "server_command": server_command,
    "model_state": model_state,
    "command": command,
  }
  exact_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
      "UPDATE runs SET finished_at = ?, error = ? WHERE agent_id = ? AND run_id = ? RETURNING *",
      (finished_at, error[:2000], agent_id, run_id),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_failed", table_name="runs", row=row)


def log_experiment(
  conn: sqlite3.Connection,
  *,
  agent_id: int,
  intent_key: str,
  metrics: Mapping[str, float | int | None] | None = None,
) -> int:
  """Atomically save benchmark metrics and mark the run successful."""
  metrics = metrics or {}
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

  values = [normalized_metrics.get(column) for column in METRIC_COLUMNS]
  intent_embedding = embed_intent(intent_key)
  finished_at = datetime.now(UTC).isoformat(timespec="seconds")

  with conn:
    # Claim the unfinished run and persist its experiment as one transaction.
    row = conn.execute(
      "UPDATE runs SET finished_at = ? WHERE agent_id = ? AND run_id = ? AND finished_at IS NULL RETURNING *",
      (finished_at, agent_id, run_id),
    ).fetchone()
    if row is None:
      raise ValueError(f"unknown or finished run: agent_id={agent_id} run_id={run_id}")
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="run_succeeded", table_name="runs", row=row)
    row = conn.execute(
      f"""
      INSERT INTO experiments(agent_id, run_id, intent_key, intent_embedding, {_METRIC_COLUMNS_SQL})
      VALUES (?, ?, ?, ?, {_METRIC_PLACEHOLDERS_SQL})
      RETURNING *
      """,
      (agent_id, run_id, intent_key, json.dumps(intent_embedding, separators=(",", ":")), *values),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="experiment_logged", table_name="experiments", row=row)
  return run_id


def find_similar_experiments(conn: sqlite3.Connection, intent: str, limit: int = 5) -> list[dict[str, object]]:
  """Return logged experiments closest to an intent embedding."""
  query_embedding = embed_intent(intent)
  rows = conn.execute(
    f"""
    SELECT e.agent_id, e.run_id, e.intent_key, e.intent_embedding, {_METRIC_SELECT_SQL},
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
    row = conn.execute(
      "INSERT INTO anomalies(agent_id, run_id, date, summary) VALUES (?, ?, ?, ?) RETURNING *",
      (agent_id, run_id, date, summary),
    ).fetchone()
    notify_main(conn, agent_id=agent_id, run_id=run_id, kind="anomaly_logged", table_name="anomalies", row=row)
  return int(row["anomaly_id"])
