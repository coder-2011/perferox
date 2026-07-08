PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agents (
  agent_id INTEGER PRIMARY KEY CHECK(agent_id >= 0),
  kind TEXT NOT NULL,
  goal TEXT NOT NULL DEFAULT '',
  experiment_cap INTEGER NOT NULL CHECK(experiment_cap >= 0),
  attempt_cap INTEGER CHECK(attempt_cap IS NULL OR attempt_cap >= 0),
  stop_requested INTEGER NOT NULL DEFAULT 0 CHECK(stop_requested IN (0, 1)),
  created_at TEXT NOT NULL
);

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
  PRIMARY KEY(agent_id, run_id),
  FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
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

CREATE INDEX IF NOT EXISTS idx_runs_exact_hash ON runs(exact_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_intent_key ON experiments(intent_key);
CREATE INDEX IF NOT EXISTS idx_anomalies_date ON anomalies(date);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON doc_chunks(source);
