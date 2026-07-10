"""Main-agent graph and tools for coordinating Perferox exploration."""

# ruff: noqa: BLE001

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox import db
from perferox.auth import write_cloud_key
from perferox.semantic import document_index
from perferox.tools import WEB_SEARCH_TOOL, run_local_command, search_files_tool

MAX_ACTIVE_SUBAGENTS = 3
MAX_EXPLORER_LINE_CHARS = 120
MAX_FILE_LINES = 240
SQL_ROW_LIMIT = 100
SUBAGENT_SESSION_PREFIX = "perferox-agent-"

MAIN_AGENT_PROMPT = """\
You are the Perferox main exploration agent.

Your job is to investigate SGLang and related systems like a performance hacker:
read code, search docs, inspect SQLite, form hypotheses, and delegate bounded
benchmark workers.

Your repository root is a persistent full clone of upstream SGLang. Preserve
existing edits and use normal Git operations when you need another branch.

ExplorerState is injected into your context every turn. Treat it as the compact
map of explored territory. Add one short line when you learn something that
should affect future exploration. Do not log private reasoning; log conclusions,
evidence, and implications. Use ExplorerState to avoid repeating experiments,
ideas, and weakly different variants.

Subagents are tools. Delegate with one rich goal and an attempt cap. Started
benchmark runs count against the cap whether they pass or fail. Do not ask for
human approval inside the configured run bounds.

Use live web search when useful for current repositories, commits, issues, and
documentation.
"""


class MainAgentState(TypedDict, total=False):
  """Graph-safe state for the main coordinator."""

  objective: str
  messages: Annotated[list[AnyMessage], add_messages]


def build_main_agent_graph(
  model: BaseChatModel,
  db_path: str | Path,
  *,
  cloud_provider: str,
  cloud_api_key: str,
  cwd: str | Path = ".",
  runtime_cwd: str | Path | None = None,
  trace_dir: str | Path = "traces",
  extra_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
  """Compile the main coordinator graph with ExplorerState hydration."""
  root = Path(cwd).resolve()
  # Keep Perferox process launches separate from the SGLang source workspace.
  runtime_root = Path(runtime_cwd).resolve() if runtime_cwd is not None else root
  traces = Path(trace_dir)
  traces = traces if traces.is_absolute() else (runtime_root / traces).resolve()
  database = Path(db_path)
  database = database if database.is_absolute() else (runtime_root / database).resolve()

  with db.open_db(database) as conn:
    db.init_db(conn)

  def refresh_sessions(conn) -> None:
    """Mark running tmux sessions missing when tmux no longer has them."""
    tmux = shutil.which("tmux")
    rows = conn.execute("SELECT session_name, agent_id, trace_ref FROM agent_sessions WHERE status IN ('running', 'ending')").fetchall()
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
    return run_local_command(command, timeout_s, cwd=root)

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
    start = start_line - 1
    try:
      with file_path.open(encoding="utf-8", errors="replace") as file:
        lines = (line.rstrip("\r\n") for line in islice(file, start, start + line_count))
        return "\n".join(f"{line_number}: {line}" for line_number, line in enumerate(lines, start_line))
    except OSError as exc:
      return f"read_file failed: {type(exc).__name__}: {exc}"

  @tool("query_sql", description="Run read-only SQLite against the Perferox database and return rows when available.")
  def query_sql(query: str, row_limit: int = SQL_ROW_LIMIT) -> str:
    """Run SQL through a read-only SQLite connection."""
    if not query.strip():
      return "query_sql failed: empty query"
    if row_limit < 1 or row_limit > SQL_ROW_LIMIT:
      return f"row_limit must be between 1 and {SQL_ROW_LIMIT}"
    try:
      with db.open_db(database, readonly=True) as conn:
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
      scored = document_index().search(db.embed_intent(query), limit)
    except Exception as exc:
      return f"query_sglang_docs failed: {type(exc).__name__}: {exc}"
    if not scored:
      return "no SGLang doc chunks ingested"
    lines = []
    for score, (source, title, url, text) in scored:
      lines.append(f"score={score:.3f} {title or source} {url or source}\n{_shorten(text, 500)}")
    return "\n\n".join(lines)

  @tool("query_intent_embeddings", description="Find logged experiments with semantically similar intent keys.")
  def query_intent_embeddings(intent: str, limit: int = 5) -> str:
    """Search experiment intent embeddings without exposing the raw vectors."""
    if limit < 1 or limit > 10:
      return "limit must be between 1 and 10"
    if not intent.strip():
      return "intent must not be empty"
    try:
      with db.open_db(database, readonly=True) as conn:
        matches = db.find_similar_experiments(conn, intent, limit)
    except Exception as exc:
      return f"query_intent_embeddings failed: {type(exc).__name__}: {exc}"
    return json.dumps(matches, indent=2, default=str) if matches else "no logged experiment intent embeddings"

  @tool("read_explorer_state", description="Read all compact ExplorerState lines.")
  def read_explorer_state() -> str:
    """Return the full compact ExplorerState ledger."""
    with db.open_db(database, readonly=True) as conn:
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
    with db.open_db(database) as conn:
      line_id = db.append_explorer_state(conn, agent_id=agent_id, line=line)
    return f"wrote ExplorerState line X{line_id:04d}"

  @tool("delegate_benchmark_subagent", description="Start one benchmark subagent for an exact repository commit, goal, and attempt cap.")
  def delegate_benchmark_subagent(repository: str, commit: str, goal: str, attempt_cap: int) -> str:
    """Start one tmux-wrapped benchmark subagent for an exact revision."""
    if attempt_cap < 1:
      return "attempt_cap must be >= 1"
    repository = repository.strip()
    commit = commit.strip()
    goal = goal.strip()
    if not repository or not commit or not goal:
      return "repository, commit, and goal must not be empty"
    tmux = shutil.which("tmux")
    if tmux is None:
      return "tmux is not installed or not on PATH"
    with db.open_db(database) as conn:
      refresh_sessions(conn)
      try:
        agent_id = db.reserve_subagent(conn, active_cap=MAX_ACTIVE_SUBAGENTS)
      except ValueError as exc:
        return str(exc)
    trace_path = traces / f"agent-{agent_id}.jsonl"
    goal_path = traces / f"agent-{agent_id}.goal.txt"
    session_name = f"{SUBAGENT_SESSION_PREFIX}{agent_id}"
    try:
      traces.mkdir(parents=True, exist_ok=True)
      trace_path.touch(exist_ok=False)
      goal_path.write_text(goal, encoding="utf-8")
      key_path = write_cloud_key(cloud_api_key)
    except OSError:
      with db.open_db(database) as conn:
        db.finish_agent_session(conn, session_name=session_name, status="missing")
      raise
    command = shlex.join([
      "uv", "run", "python", "-m", "perferox.process_host", "subagent",
      "--agent-id", str(agent_id), "--db-path", str(database),
      "--trace-path", str(trace_path), "--goal-file", str(goal_path),
      "--repository", repository, "--commit", commit,
      "--cloud-key-file", str(key_path),
      "--attempt-cap", str(attempt_cap),
    ])
    try:
      result = subprocess.run(
        [tmux, "new-session", "-d", "-s", session_name, "-c", str(runtime_root), "--", "bash", "-lc", command],
        text=True,
        capture_output=True,
        check=False,
      )
    except OSError:
      # Delete a secret handoff that tmux never delivered.
      key_path.unlink(missing_ok=True)
      with db.open_db(database) as conn:
        db.finish_agent_session(conn, session_name=session_name, status="missing")
      raise
    if result.returncode != 0:
      key_path.unlink(missing_ok=True)
      with db.open_db(database) as conn:
        db.finish_agent_session(conn, session_name=session_name, status="missing")
      return f"subagent tmux launch failed: {(result.stderr or result.stdout).strip()}"
    with db.open_db(database) as conn:
      db.activate_subagent(conn, agent_id=agent_id, trace_ref=str(trace_path))
    line = _shorten(f"start agent-{agent_id} {repository}@{commit} attempts={attempt_cap}: {goal}", MAX_EXPLORER_LINE_CHARS)
    with db.open_db(database) as conn:
      db.append_explorer_state(conn, agent_id=None, line=line)
    return f"started agent_id={agent_id} repository={repository} commit={commit} attempt_cap={attempt_cap} session={session_name} trace={trace_path} attach='tmux attach -t {session_name}'"

  tools = [
    bash,
    search_files_tool(root),
    read_file,
    query_sql,
    query_sglang_docs,
    query_intent_embeddings,
    read_explorer_state,
    write_explorer_state,
    delegate_benchmark_subagent,
    *extra_tools,
  ]
  # Native web search executes server-side and never enters the local ToolNode.
  bound_model = model.bind_tools([*tools, WEB_SEARCH_TOOL], parallel_tool_calls=True)

  def call_model(state: MainAgentState) -> dict[str, list[BaseMessage]]:
    """Invoke the main model with fresh ExplorerState in context."""
    with db.open_db(database) as conn:
      refresh_sessions(conn)
      lines = db.read_explorer_state(conn)
      session_rows = conn.execute("SELECT * FROM agent_sessions ORDER BY rowid DESC LIMIT 8").fetchall()
    objective = state.get("objective", "") or "(none)"
    explorer_state = "\n".join(lines) if lines else "(empty)"
    sessions = json.dumps([dict(row) for row in session_rows], default=str) if session_rows else "(none)"
    system_prompt = f"{MAIN_AGENT_PROMPT}\n\nCloud provider: {cloud_provider}\n\nExplorerState:\n{explorer_state}\n\nTmuxSessions:\n{sessions}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=objective), *state.get("messages", [])]
    return {"messages": [bound_model.invoke(messages)]}

  def route_after_main(state: MainAgentState) -> Literal["tools", "__end__"]:
    """Stop before another tool call once the host accepted End."""
    if not getattr(state["messages"][-1], "tool_calls", None):
      return END
    with db.open_db(database, readonly=True) as conn:
      row = conn.execute("SELECT status FROM agent_sessions WHERE session_name = ?", ("perferox-main",)).fetchone()
    return END if row is not None and row["status"] == "ending" else "tools"

  graph = StateGraph(MainAgentState)
  graph.add_node("main", call_model)
  graph.add_node("tools", ToolNode(tools, name="main_tools"))
  graph.add_edge(START, "main")
  graph.add_conditional_edges("main", route_after_main)
  graph.add_edge("tools", "main")
  return graph.compile(name="perferox_main_agent")


def _shorten(text: str, limit: int = 80) -> str:
  """Shorten text for compact ExplorerState lines."""
  compact = " ".join(text.split())
  if len(compact) <= limit:
    return compact
  return compact[:limit - 1].rstrip() + "..."
