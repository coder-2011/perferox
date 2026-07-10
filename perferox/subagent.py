"""LangGraph lifecycle and JSONL tracing for one benchmark subagent."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox import db
from perferox.prompts import BENCHMARK_SYSTEM_PROMPT, CREATE_POD_SYSTEM_PROMPT, SETUP_SYSTEM_PROMPT
from perferox.remote import SessionRegistry
from perferox.tools import (
  WEB_SEARCH_TOOL,
  connect_remote_session,
  local_terminal,
  log_anomaly_tool,
  log_experiment_tool,
  remote_terminal,
  sglang_bench_serving,
)


class SubagentState(TypedDict, total=False):
  """Graph-safe state for one benchmark subagent."""

  agent_id: int
  objective: str
  messages: Annotated[list[AnyMessage], add_messages]
  summary: str


def trace_jsonable(value: Any) -> Any:
  """Convert trace payload leftovers into JSON-safe values."""
  if isinstance(value, Path):
    return str(value)
  if hasattr(value, "model_dump"):
    return value.model_dump()
  return repr(value)


def stream_with_trace(
  graph: Any,
  state: Mapping[str, Any],
  path: str | Path,
) -> Iterator[Any]:
  """Stream graph updates and append them to one JSONL trace."""
  agent_id = state.get("agent_id")
  trace_file = Path(path)
  trace_file.parent.mkdir(parents=True, exist_ok=True)
  with trace_file.open("a", encoding="utf-8", buffering=1) as file:
    for event in graph.stream(state, stream_mode="updates"):
      record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_id": agent_id,
        "kind": "graph_update",
        "payload": event,
      }
      file.write(json.dumps(record, separators=(",", ":"), default=trace_jsonable) + "\n")
      yield event


def _model_node(model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str) -> Callable[[SubagentState], dict[str, list[BaseMessage]]]:
  """Return a LangGraph node that invokes one chat model step."""
  # Native web search completes inside the model call; ToolNode handles local tools.
  bound_model = model.bind_tools([*tools, WEB_SEARCH_TOOL], parallel_tool_calls=False)

  def call_model(state: SubagentState) -> dict[str, list[BaseMessage]]:
    """Invoke the model with the subagent goal in the system prompt."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.get("objective", "(none)")), *state.get("messages", [])]
    return {"messages": [bound_model.invoke(messages)]}

  return call_model


def _message_text(message: BaseMessage) -> str:
  """Extract readable text from a LangChain message."""
  content = message.content
  if isinstance(content, str):
    return content
  if isinstance(content, Sequence):
    parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
    return "\n".join(part for part in parts if part)
  return repr(content)


def build_subagent_graph(
  model: BaseChatModel,
  agent_id: int,
  session_registry: SessionRegistry,
  db_path: str | Path,
  repository: str,
  commit: str,
  *,
  create_pod_prompt: str = CREATE_POD_SYSTEM_PROMPT,
  attempt_cap: int = 1,
  trace_ref: str = "",
  create_pod_tools: Sequence[BaseTool] = (local_terminal,),
  setup_tools: Sequence[BaseTool] = (),
  benchmark_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
  """Compile the fixed subagent lifecycle graph."""
  session_id = f"agent-{agent_id}"
  target_prompt = f"\n\nTarget repository:\n{repository}\n\nTarget commit:\n{commit}"
  graph = StateGraph(SubagentState)
  remote_tool = remote_terminal(session_registry, session_id)
  create_pod_tools = [*create_pod_tools, connect_remote_session(session_registry, session_id)]
  setup_tools = [*setup_tools, remote_tool]
  benchmark_tools = [
    *benchmark_tools,
    remote_tool,
    sglang_bench_serving(session_registry, session_id, db_path, agent_id, trace_ref, attempt_cap),
    log_experiment_tool(db_path, agent_id),
    log_anomaly_tool(db_path, agent_id),
  ]
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)

  def runtime_status() -> tuple[bool, int]:
    """Read the host-owned stop flag and started-attempt count together."""
    with closing(db.connect(db_path, readonly=True)) as conn:
      stopped = db.stop_requested(conn, agent_id=agent_id)
      attempts = conn.execute("SELECT COUNT(*) FROM runs WHERE agent_id = ?", (agent_id,)).fetchone()[0]
    return stopped, int(attempts)

  def route_after_create_pod(state: SubagentState) -> Literal["create_pod_tools", "basic_setup", "wrap_up"]:
    """Prevent another provisioning action after a soft stop."""
    stopped, _ = runtime_status()
    if stopped:
      return "wrap_up"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "create_pod_tools"
    return "basic_setup"

  def route_after_basic_setup(state: SubagentState) -> Literal["basic_setup_tools", "setup_intervention", "benchmark_loop", "wrap_up"]:
    """Choose setup tools, intervention, benchmark, or wrap-up."""
    stopped, attempts = runtime_status()
    if stopped or attempts >= attempt_cap:
      return "wrap_up"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "basic_setup_tools"
    if "setup_failed" in _message_text(last_message).lower():
      return "setup_intervention"
    return "benchmark_loop"

  def route_after_setup_intervention(state: SubagentState) -> Literal["setup_intervention_tools", "basic_setup", "wrap_up"]:
    """Route setup intervention back to setup or out to wrap-up."""
    stopped, _ = runtime_status()
    if stopped:
      return "wrap_up"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "setup_intervention_tools"
    if "setup_failed" in _message_text(last_message).lower():
      return "wrap_up"
    return "basic_setup"

  def route_after_benchmark(state: SubagentState) -> Literal["benchmark_tools", "wrap_up"]:
    """Keep benchmarking until the started-attempt cap is reached."""
    stopped, attempts = runtime_status()
    if stopped or attempts >= attempt_cap:
      return "wrap_up"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "benchmark_tools"
    return "wrap_up"

  def wrap_up(state: SubagentState) -> dict[str, Any]:
    """Generate and notify the main agent with a final worker summary."""
    state_messages = state.get("messages", [])
    objective = state.get("objective", "(none)")
    summary_prompt = (
      "Close out this Perferox benchmark subagent with a concise factual summary. "
      "Include what was attempted, useful run IDs, anomalies, blockers, and the best next step.\n\n"
      f"Agent: {agent_id}\nRepository: {repository}\nCommit: {commit}\nObjective:\n{objective}\nLoop cap: {attempt_cap}"
    )
    response = model.invoke([SystemMessage(content=summary_prompt), *state_messages])
    summary = _message_text(response)
    _, started_attempts = runtime_status()
    row = {
      "agent_id": agent_id,
      "repository": repository,
      "commit": commit,
      "objective": objective,
      "summary": summary,
      "started_attempts": started_attempts,
      "loop_cap": attempt_cap,
      "trace_ref": trace_ref,
    }
    with closing(db.connect(db_path)) as conn, conn:
      db.notify_main(conn, agent_id=agent_id, run_id=None, kind="subagent_summary", table_name="subagent_summary", row=row)
    return {"summary": summary, "messages": [response]}

  for name, tool_node, tools, prompt in (
    ("create_pod", "create_pod_tools", create_pod_tools, create_pod_prompt),
    ("basic_setup", "basic_setup_tools", setup_tools, SETUP_SYSTEM_PROMPT),
    ("setup_intervention", "setup_intervention_tools", setup_tools, SETUP_SYSTEM_PROMPT),
    ("benchmark_loop", "benchmark_tools", benchmark_tools, BENCHMARK_SYSTEM_PROMPT),
  ):
    phase_prompt = f"{prompt}{target_prompt}\n\nHard cap: start at most {attempt_cap} benchmark attempt(s), then summarize."
    graph.add_node(name, _model_node(model, tools, phase_prompt))
    graph.add_node(tool_node, ToolNode(tools, name=tool_node))
  graph.add_node("wrap_up", wrap_up)

  graph.add_edge(START, "create_pod")
  graph.add_conditional_edges("create_pod", route_after_create_pod)
  graph.add_edge("create_pod_tools", "create_pod")
  graph.add_conditional_edges("basic_setup", route_after_basic_setup)
  graph.add_edge("basic_setup_tools", "basic_setup")
  graph.add_conditional_edges("setup_intervention", route_after_setup_intervention)
  graph.add_edge("setup_intervention_tools", "setup_intervention")
  graph.add_conditional_edges("benchmark_loop", route_after_benchmark)
  graph.add_edge("benchmark_tools", "benchmark_loop")
  graph.add_edge("wrap_up", END)
  return graph.compile(name="perferox_subagent")
