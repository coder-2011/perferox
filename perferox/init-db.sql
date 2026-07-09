PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  agent_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL CHECK(run_id >= 0),
  gpu TEXT NOT NULL DEFAULT '',
  started_at TEXT NOT NULL,
  finished_at TEXT,
  trace_ref TEXT,
  command TEXT NOT NULL DEFAULT '',
  exact_hash TEXT NOT NULL UNIQUE,
  error TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(agent_id, run_id)
);

CREATE TABLE IF NOT EXISTS experiments (
  agent_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  intent_key TEXT NOT NULL,
  intent_embedding TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS explorer_state_lines (
  line_id INTEGER PRIMARY KEY,
  agent_id INTEGER,
  created_at TEXT NOT NULL,
  line TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_sessions (
  session_name TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  agent_id INTEGER,
  status TEXT NOT NULL,
  trace_ref TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS main_notifications (
  notification_id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  delivered_at TEXT,
  agent_id INTEGER,
  run_id INTEGER,
  kind TEXT NOT NULL,
  table_name TEXT NOT NULL,
  row_json TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_experiments_intent_key ON experiments(intent_key);
CREATE INDEX IF NOT EXISTS idx_anomalies_date ON anomalies(date);
CREATE INDEX IF NOT EXISTS idx_explorer_state_lines_created_at ON explorer_state_lines(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
CREATE INDEX IF NOT EXISTS idx_main_notifications_delivered ON main_notifications(delivered_at, notification_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON doc_chunks(source);
