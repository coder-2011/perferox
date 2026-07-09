"""Main-agent graph and tools for coordinating Perferox exploration."""

# ruff: noqa: BLE001

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox import db
from perferox.remote import SessionRegistry
from perferox.subagent import SubagentState, build_subagent_graph, stream_with_trace
from perferox.tools import search_files_tool

MAX_ACTIVE_SUBAGENTS = 3
MAX_EXPLORER_LINE_CHARS = 120
MAX_FILE_LINES = 240
MAX_OUTPUT_CHARS = 10000
SQL_ROW_LIMIT = 100

MAIN_AGENT_PROMPT = """\
You are the Perferox main exploration agent.

Your job is to investigate SGLang and related systems like a performance hacker:
read code, search docs, inspect SQLite, form hypotheses, and delegate bounded
benchmark workers.

ExplorerState is injected into your context every turn. Treat it as the compact
map of explored territory. Add one short line when you learn something that
should affect future exploration. Do not log private reasoning; log conclusions,
evidence, and implications. Use ExplorerState to avoid repeating experiments,
ideas, and weakly different variants.

Subagents are tools. Delegate with one rich goal and an attempt cap. Started
benchmark runs count against the cap whether they pass or fail. Do not ask for
human approval inside the configured run bounds.
"""


class MainAgentState(TypedDict, total=False):
  """Graph-safe state for the main coordinator."""

  objective: str
  messages: Annotated[list[AnyMessage], add_messages]


def build_main_agent_graph(
  model: BaseChatModel,
  db_path: str | Path,
  *,
  cwd: str | Path = ".",
  trace_dir: str | Path = "traces",
  extra_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
  """Compile the main coordinator graph with ExplorerState hydration."""
  root = Path(cwd).resolve()
  traces = Path(trace_dir)
  session_registry = SessionRegistry()
  active_subagents: dict[int, threading.Thread] = {}

  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)

  @tool("bash", description="Run one local bash command from the repository root.")
  def bash(command: str, timeout_s: float = 30.0) -> str:
    """Run one shell command and return bounded output."""
    try:
      process = subprocess.Popen(
        ["bash", "-lc", command],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=root,
        start_new_session=os.name == "posix",
      )
      stdout, stderr = process.communicate(timeout=timeout_s)
      exit_status = process.returncode
    except subprocess.TimeoutExpired:
      os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
      stdout, stderr = process.communicate()
      exit_status = None
      stderr = f"{stderr}\ntimed out after {timeout_s}s"
    except Exception as exc:
      stdout, stderr = "", f"{type(exc).__name__}: {exc}"
      exit_status = None
    output = "\n".join(part for part in (stdout, stderr) if part)
    if len(output) > MAX_OUTPUT_CHARS:
      keep = MAX_OUTPUT_CHARS // 2
      output = f"{output[:keep]}\n\n... {len(output) - MAX_OUTPUT_CHARS} chars elided ...\n\n{output[-keep:]}"
    return f"exit_code={exit_status}\n{output}"

  @tool("read_file", description="Read a repository file with optional 1-based line range.")
  def read_file(path: str, start_line: int = 1, line_count: int = MAX_FILE_LINES) -> str:
    """Read a bounded slice of one repository file."""
    if start_line < 1:
      return "start_line must be >= 1"
    if line_count < 1 or line_count > MAX_FILE_LINES:
      return f"line_count must be between 1 and {MAX_FILE_LINES}"
    try:
      file_path = (root / path).resolve()
      file_path.relative_to(root)
    except ValueError:
      return f"path escapes repository root: {path}"
    try:
      lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
      return f"read_file failed: {type(exc).__name__}: {exc}"
    start = start_line - 1
    selected = lines[start:start + line_count]
    return "\n".join(f"{start + index + 1}: {line}" for index, line in enumerate(selected))

  @tool("query_sql", description="Run one read-only SQLite SELECT/WITH query against the Perferox database.")
  def query_sql(query: str, row_limit: int = SQL_ROW_LIMIT) -> str:
    """Run a bounded read-only SQL query."""
    stripped = query.strip()
    lowered = stripped.casefold()
    blocked = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "detach", "pragma", "vacuum")
    if row_limit < 1 or row_limit > SQL_ROW_LIMIT:
      return f"row_limit must be between 1 and {SQL_ROW_LIMIT}"
    if not lowered.startswith(("select ", "with ")):
      return "only SELECT/WITH queries are allowed"
    if ";" in stripped.rstrip(";"):
      return "only one SQL statement is allowed"
    if any(re.search(rf"\b{token}\b", lowered) for token in blocked):
      return "query contains a blocked SQL keyword"
    try:
      with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        rows = conn.execute(query).fetchmany(row_limit)
    except Exception as exc:
      return f"query_sql failed: {type(exc).__name__}: {exc}"
    return json.dumps([dict(row) for row in rows], indent=2, default=str)

  @tool("query_sglang_docs", description="Search locally ingested SGLang docs chunks by semantic similarity.")
  def query_sglang_docs(query: str, limit: int = 5) -> str:
    """Search doc chunks with the existing embedding path."""
    if limit < 1 or limit > 10:
      return "limit must be between 1 and 10"
    try:
      query_embedding = db.embed_intent(query)
      with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        rows = conn.execute("SELECT source, title, url, text, embedding FROM doc_chunks").fetchall()
    except Exception as exc:
      return f"query_sglang_docs failed: {type(exc).__name__}: {exc}"
    scored = [
      (sum(a * b for a, b in zip(query_embedding, json.loads(row["embedding"]))), row)
      for row in rows
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
      return "no SGLang doc chunks ingested"
    lines = []
    for score, row in scored[:limit]:
      title = row["title"] or row["source"]
      location = row["url"] or row["source"]
      lines.append(f"score={score:.3f} {title} {location}\n{_shorten(str(row['text']), 500)}")
    return "\n\n".join(lines)

  @tool("read_explorer_state", description="Read all compact ExplorerState lines.")
  def read_explorer_state() -> str:
    """Return the full compact ExplorerState ledger."""
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      lines = db.read_explorer_state(conn)
    return "\n".join(lines) if lines else "(empty)"

  @tool("write_explorer_state", description="Append exactly one compact ExplorerState line, ideally 10-15 high-quality words.")
  def write_explorer_state(line: str, agent_id: int | None = None) -> str:
    """Append one compact factual line to ExplorerState."""
    line = " ".join(line.split())
    if not line:
      return "empty ExplorerState line refused"
    if len(line) > MAX_EXPLORER_LINE_CHARS:
      return f"ExplorerState line too long ({len(line)}/{MAX_EXPLORER_LINE_CHARS} chars)"
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      line_id = db.append_explorer_state(conn, agent_id=agent_id, line=line)
    return f"wrote ExplorerState line X{line_id:04d}"

  @tool("delegate_benchmark_subagent", description="Start one benchmark subagent with a rich goal and started-attempt cap.")
  def delegate_benchmark_subagent(goal: str, attempt_cap: int) -> str:
    """Start one background benchmark subagent."""
    if attempt_cap < 1:
      return "attempt_cap must be >= 1"
    for agent_id, thread in list(active_subagents.items()):
      if not thread.is_alive():
        active_subagents.pop(agent_id, None)
    if len(active_subagents) >= MAX_ACTIVE_SUBAGENTS:
      return f"max active subagents reached ({len(active_subagents)}/{MAX_ACTIVE_SUBAGENTS})"
    traces.mkdir(parents=True, exist_ok=True)
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      db_next = conn.execute("SELECT COALESCE(MAX(agent_id) + 1, 0) FROM runs").fetchone()[0]
    trace_ids = [int(path.stem[6:]) for path in traces.glob("agent-*.jsonl") if path.stem[6:].isdigit()]
    agent_id = max(db_next, max(trace_ids, default=-1) + 1)
    trace_path = traces / f"agent-{agent_id}.jsonl"
    trace_path.touch(exist_ok=False)
    graph = build_subagent_graph(
      model,
      agent_id,
      session_registry,
      db_path,
      attempt_cap=attempt_cap,
      trace_ref=str(trace_path),
    )
    state: SubagentState = {"agent_id": agent_id, "messages": [HumanMessage(content=goal)]}
    events = stream_with_trace(graph, state, trace_path, lambda: None, session_registry)

    def drain_events() -> None:
      """Consume a background subagent stream so it can run independently."""
      try:
        for _ in events:
          pass
      except Exception as exc:
        line = _shorten(f"B agent-{agent_id} failed: {type(exc).__name__} {exc}", MAX_EXPLORER_LINE_CHARS)
        with closing(db.connect(db_path)) as conn:
          db.init_db(conn)
          db.append_explorer_state(conn, agent_id=None, line=line)
      finally:
        active_subagents.pop(agent_id, None)

    thread = threading.Thread(target=drain_events, daemon=True)
    active_subagents[agent_id] = thread
    thread.start()
    line = _shorten(f"start agent-{agent_id} attempts={attempt_cap}: {goal}", MAX_EXPLORER_LINE_CHARS)
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      db.append_explorer_state(conn, agent_id=None, line=line)
    return f"started agent_id={agent_id} attempt_cap={attempt_cap} trace={trace_path}"

  tools = [
    bash,
    search_files_tool(root),
    read_file,
    query_sql,
    query_sglang_docs,
    read_explorer_state,
    write_explorer_state,
    delegate_benchmark_subagent,
    *extra_tools,
  ]
  bound_model = model.bind_tools(list(tools))

  def call_model(state: MainAgentState) -> dict[str, list[BaseMessage]]:
    """Invoke the main model with fresh ExplorerState in context."""
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      lines = db.read_explorer_state(conn)
    objective = state.get("objective", "") or "(none)"
    explorer_state = "\n".join(lines) if lines else "(empty)"
    system_prompt = f"{MAIN_AGENT_PROMPT}\n\nObjective:\n{objective}\n\nExplorerState:\n{explorer_state}"
    messages = [SystemMessage(content=system_prompt), *state.get("messages", [])]
    return {"messages": [bound_model.invoke(messages)]}

  def route_after_main(state: MainAgentState) -> str:
    """Route to tools when the model requested tool calls."""
    if getattr(state["messages"][-1], "tool_calls", None):
      return "tools"
    return END

  graph = StateGraph(MainAgentState)
  graph.add_node("main", call_model)
  graph.add_node("tools", ToolNode(tools, name="main_tools"))
  graph.add_edge(START, "main")
  graph.add_conditional_edges("main", route_after_main, {"tools": "tools", END: END})
  graph.add_edge("tools", "main")
  return graph.compile(name="perferox_main_agent")


def _shorten(text: str, limit: int = 80) -> str:
  """Shorten text for compact ExplorerState lines."""
  compact = " ".join(text.split())
  if len(compact) <= limit:
    return compact
  return compact[:limit - 1].rstrip() + "..."
