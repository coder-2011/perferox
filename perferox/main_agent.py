"""Main-agent graph and tools for coordinating Perferox exploration."""

# ruff: noqa: BLE001

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox import db
from perferox.tools import search_files_tool

MAX_ACTIVE_SUBAGENTS = 3
MAX_EXPLORER_LINE_CHARS = 120
MAX_FILE_LINES = 240
MAX_OUTPUT_CHARS = 10000
SQL_ROW_LIMIT = 100
SUBAGENT_SESSION_PREFIX = "perferox-agent-"

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
  traces = traces if traces.is_absolute() else (root / traces).resolve()
  database = Path(db_path)
  database = database if database.is_absolute() else (root / database).resolve()

  with closing(db.connect(database)) as conn:
    db.init_db(conn)

  def refresh_sessions(conn) -> None:
    """Mark running tmux sessions missing when tmux no longer has them."""
    tmux = shutil.which("tmux")
    rows = conn.execute("SELECT * FROM agent_sessions WHERE status = 'running'").fetchall()
    for row in rows:
      alive = tmux and subprocess.run(
        [tmux, "has-session", "-t", row["session_name"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
      ).returncode == 0
      if alive:
        continue
      if db.finish_agent_session(conn, session_name=row["session_name"], status="missing"):
        label = f"agent-{row['agent_id']}" if row["agent_id"] is not None else row["session_name"]
        line = _shorten(f"{label} tmux missing; trace {Path(row['trace_ref']).name}", MAX_EXPLORER_LINE_CHARS)
        db.append_explorer_state(conn, agent_id=row["agent_id"], line=line)

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

  @tool("query_sql", description="Run read-only SQLite against the Perferox database and return rows when available.")
  def query_sql(query: str, row_limit: int = SQL_ROW_LIMIT) -> str:
    """Run SQL through a read-only SQLite connection."""
    if not query.strip():
      return "query_sql failed: empty query"
    if row_limit < 1 or row_limit > SQL_ROW_LIMIT:
      return f"row_limit must be between 1 and {SQL_ROW_LIMIT}"
    try:
      with closing(db.connect(database, readonly=True)) as conn:
        cursor = conn.execute(query)
        if cursor.description is None:
          return f"ok rowcount={cursor.rowcount}"
        rows = cursor.fetchmany(row_limit)
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
      with closing(db.connect(database)) as conn:
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
    with closing(db.connect(database)) as conn:
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
    with closing(db.connect(database)) as conn:
      db.init_db(conn)
      line_id = db.append_explorer_state(conn, agent_id=agent_id, line=line)
    return f"wrote ExplorerState line X{line_id:04d}"

  @tool("delegate_benchmark_subagent", description="Start one benchmark subagent with a rich goal and started-attempt cap.")
  def delegate_benchmark_subagent(goal: str, attempt_cap: int) -> str:
    """Start one tmux-wrapped benchmark subagent process."""
    if attempt_cap < 1:
      return "attempt_cap must be >= 1"
    if not goal.strip():
      return "goal must not be empty"
    with closing(db.connect(database)) as conn:
      db.init_db(conn)
      refresh_sessions(conn)
      active_count = conn.execute("SELECT COUNT(*) FROM agent_sessions WHERE status = 'running' AND role = 'subagent'").fetchone()[0]
    if active_count >= MAX_ACTIVE_SUBAGENTS:
      return f"max active subagents reached ({active_count}/{MAX_ACTIVE_SUBAGENTS})"
    tmux = shutil.which("tmux")
    if tmux is None:
      return "tmux is not installed or not on PATH"
    traces.mkdir(parents=True, exist_ok=True)
    with closing(db.connect(database)) as conn:
      db.init_db(conn)
      db_next = conn.execute("SELECT COALESCE(MAX(agent_id) + 1, 0) FROM runs").fetchone()[0]
      session_next = conn.execute("SELECT COALESCE(MAX(agent_id) + 1, 0) FROM agent_sessions WHERE agent_id IS NOT NULL").fetchone()[0]
    trace_ids = [int(path.stem[6:]) for path in traces.glob("agent-*.jsonl") if path.stem[6:].isdigit()]
    agent_id = max(db_next, session_next, max(trace_ids, default=-1) + 1)
    trace_path = traces / f"agent-{agent_id}.jsonl"
    goal_path = traces / f"agent-{agent_id}.goal.txt"
    session_name = f"{SUBAGENT_SESSION_PREFIX}{agent_id}"
    trace_path.touch(exist_ok=False)
    goal_path.write_text(goal, encoding="utf-8")
    command = shlex.join([
      "uv", "run", "python", "-m", "perferox.agent_runner", "subagent",
      "--agent-id", str(agent_id), "--db-path", str(database),
      "--trace-path", str(trace_path), "--goal-file", str(goal_path),
      "--attempt-cap", str(attempt_cap),
    ])
    result = subprocess.run(
      [tmux, "new-session", "-d", "-s", session_name, "-c", str(root), "--", "bash", "-lc", command],
      text=True,
      capture_output=True,
      check=False,
    )
    if result.returncode != 0:
      return f"subagent tmux launch failed: {(result.stderr or result.stdout).strip()}"
    line = _shorten(f"start agent-{agent_id} attempts={attempt_cap}: {goal}", MAX_EXPLORER_LINE_CHARS)
    with closing(db.connect(database)) as conn:
      db.init_db(conn)
      db.append_explorer_state(conn, agent_id=None, line=line)
    return f"started agent_id={agent_id} attempt_cap={attempt_cap} session={session_name} trace={trace_path} attach='tmux attach -t {session_name}'"

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
    with closing(db.connect(database)) as conn:
      db.init_db(conn)
      refresh_sessions(conn)
      lines = db.read_explorer_state(conn)
      session_rows = conn.execute("SELECT * FROM agent_sessions ORDER BY rowid DESC LIMIT 8").fetchall()
    objective = state.get("objective", "") or "(none)"
    explorer_state = "\n".join(lines) if lines else "(empty)"
    sessions = json.dumps([dict(row) for row in session_rows], default=str) if session_rows else "(none)"
    system_prompt = f"{MAIN_AGENT_PROMPT}\n\nObjective:\n{objective}\n\nExplorerState:\n{explorer_state}\n\nTmuxSessions:\n{sessions}"
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
