"""Main-agent coordinator module for Perferox."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from perferox import db
from perferox.remote import SessionRegistry
from perferox.subagent import SubagentState, build_subagent_graph, stream_with_trace


def start_benchmark_subagent(
  model: BaseChatModel,
  db_path: str | Path,
  goal: str,
  teardown: Callable[[], None],
  experiment_cap: int = 1,
  trace_dir: str | Path = "traces",
  session_registry: SessionRegistry | None = None,
) -> tuple[int, Path, Iterator[Any]]:
  """Create one benchmark worker and return its traced event stream."""
  registry = session_registry or SessionRegistry()
  trace_root = Path(trace_dir)
  trace_root.mkdir(parents=True, exist_ok=True)
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    agent_id = _next_agent_id(conn, trace_root)
  trace_path = trace_root / f"agent-{agent_id}.jsonl"
  trace_path.touch(exist_ok=False)
  graph = build_subagent_graph(
    model,
    agent_id,
    registry,
    db_path,
    experiment_cap=experiment_cap,
    trace_ref=str(trace_path),
  )
  state: SubagentState = {"agent_id": agent_id, "messages": [HumanMessage(content=goal)]}
  events = stream_with_trace(graph, state, trace_path, teardown, registry)
  return agent_id, trace_path, events


def _next_agent_id(conn, trace_root: Path) -> int:
  """Return the next id without keeping an agents table."""
  db_next = conn.execute("SELECT COALESCE(MAX(agent_id) + 1, 0) FROM runs").fetchone()[0]
  trace_ids = [int(path.stem[6:]) for path in trace_root.glob("agent-*.jsonl") if path.stem[6:].isdigit()]
  return max(db_next, max(trace_ids, default=-1) + 1)
