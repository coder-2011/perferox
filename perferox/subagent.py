"""LangGraph lifecycle and JSONL tracing for one benchmark subagent."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox import db
from perferox.prompts import BENCHMARK_SYSTEM_PROMPT, CREATE_POD_SYSTEM_PROMPT, SETUP_SYSTEM_PROMPT
from perferox.remote import SessionRegistry
from perferox.tools import (
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
  loop_cap: int
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
  with trace_file.open("a", encoding="utf-8") as file:
    for event in graph.stream(state, stream_mode="updates"):
      record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_id": agent_id,
        "kind": "graph_update",
        "payload": event,
      }
      file.write(json.dumps(record, separators=(",", ":"), default=trace_jsonable) + "\n")
      file.flush()
      yield event


def _model_node(model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str) -> Callable[[SubagentState], dict[str, list[BaseMessage]]]:
  """Return a LangGraph node that invokes one chat model step."""
  bound_model = model.bind_tools(list(tools)) if tools else model

  def call_model(state: SubagentState) -> dict[str, list[BaseMessage]]:
    """Invoke the model with the subagent goal in the system prompt."""
    state_messages = state.get("messages", [])
    objective = _message_text(state_messages[0]) if state_messages else "(none)"
    loop_cap = state.get("loop_cap", "(none)")
    messages = [SystemMessage(content=f"{system_prompt}\n\nObjective:\n{objective}\n\nLoop cap:\n{loop_cap}"), *state_messages]
    response = bound_model.invoke(messages)
    return {"messages": [response]}

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


def _started_attempts(state: SubagentState) -> int:
  """Count benchmark attempts that actually started."""
  return sum(
    getattr(message, "type", "") == "tool"
    and getattr(message, "name", "") == "sglang_bench_serving"
    and str(getattr(message, "content", "")).startswith("run_id=")
    for message in state.get("messages", [])
  )


def _stop_requested(db_path: str | Path, agent_id: int) -> bool:
  """Check the host-owned stop flag before starting more benchmark work."""
  with closing(db.connect(db_path)) as conn:
    db.init_db(conn)
    return db.stop_requested(conn, agent_id=agent_id)


def build_subagent_graph(
  model: BaseChatModel,
  agent_id: int,
  session_registry: SessionRegistry,
  db_path: str | Path,
  attempt_cap: int = 1,
  trace_ref: str = "",
  create_pod_tools: Sequence[BaseTool] = (local_terminal,),
  setup_tools: Sequence[BaseTool] = (),
  benchmark_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
  """Compile the fixed subagent lifecycle graph."""
  session_id = f"agent-{agent_id}"
  graph = StateGraph(SubagentState)
  create_tool_list = list(create_pod_tools)
  setup_tool_list = list(setup_tools)
  benchmark_tool_list = list(benchmark_tools)
  benchmark_prompt = (
    BENCHMARK_SYSTEM_PROMPT
    + f"\nHard cap: start at most {attempt_cap} benchmark attempt(s), then summarize."
  )
  remote_tool = remote_terminal(session_registry, session_id)
  create_tool_list.append(connect_remote_session(session_registry, session_id))
  setup_tool_list.append(remote_tool)
  benchmark_tool_list.append(remote_tool)
  benchmark_tool_list.append(sglang_bench_serving(session_registry, session_id, db_path, agent_id, trace_ref, attempt_cap))
  benchmark_tool_list.append(log_experiment_tool(db_path, agent_id))
  benchmark_tool_list.append(log_anomaly_tool(db_path, agent_id))

  def route_after_create_pod(state: SubagentState) -> str:
    """Route pod creation through tools until SSH details are ready."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "create_pod_tools"
    return "basic_setup"

  def route_after_basic_setup(state: SubagentState) -> str:
    """Choose setup tools, intervention, benchmark, or wrap-up."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "basic_setup_tools"
    if "setup_failed" in _message_text(last_message).lower():
      return "setup_intervention"
    if _stop_requested(db_path, agent_id) or _started_attempts(state) >= attempt_cap:
      return "wrap_up"
    return "benchmark_loop"

  def route_after_setup_intervention(state: SubagentState) -> str:
    """Route setup intervention back to setup or out to wrap-up."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "setup_intervention_tools"
    if "setup_failed" in _message_text(last_message).lower():
      return "wrap_up"
    return "basic_setup"

  def route_after_benchmark(state: SubagentState) -> str:
    """Keep benchmarking until the started-attempt cap is reached."""
    if _stop_requested(db_path, agent_id) or _started_attempts(state) >= attempt_cap:
      return "wrap_up"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
      return "benchmark_tools"
    return "wrap_up"

  def wrap_up(state: SubagentState) -> dict[str, Any]:
    """Generate and notify the main agent with a final worker summary."""
    state_messages = state.get("messages", [])
    objective = _message_text(state_messages[0]) if state_messages else "(none)"
    summary_prompt = (
      "Close out this Perferox benchmark subagent with a concise factual summary. "
      "Include what was attempted, useful run IDs, anomalies, blockers, and the best next step.\n\n"
      f"Agent: {agent_id}\nObjective:\n{objective}\nLoop cap: {state.get('loop_cap', attempt_cap)}"
    )
    response = model.invoke([SystemMessage(content=summary_prompt), *state_messages])
    summary = _message_text(response)
    row = {
      "agent_id": agent_id,
      "objective": objective,
      "summary": summary,
      "started_attempts": _started_attempts(state),
      "loop_cap": state.get("loop_cap", attempt_cap),
      "trace_ref": trace_ref,
    }
    with closing(db.connect(db_path)) as conn:
      db.init_db(conn)
      with conn:
        db.notify_main(conn, agent_id=agent_id, run_id=None, kind="subagent_summary", table_name="subagent_summary", row=row)
    return {"summary": summary, "messages": [response]}

  for name, tool_node, tools, prompt in (
    ("create_pod", "create_pod_tools", create_tool_list, CREATE_POD_SYSTEM_PROMPT),
    ("basic_setup", "basic_setup_tools", setup_tool_list, SETUP_SYSTEM_PROMPT),
    ("setup_intervention", "setup_intervention_tools", setup_tool_list, SETUP_SYSTEM_PROMPT),
    ("benchmark_loop", "benchmark_tools", benchmark_tool_list, benchmark_prompt),
  ):
    graph.add_node(name, _model_node(model, tools, prompt))
    graph.add_node(tool_node, ToolNode(tools, name=tool_node))
  graph.add_node("wrap_up", wrap_up)

  graph.add_edge(START, "create_pod")
  graph.add_conditional_edges(
    "create_pod",
    route_after_create_pod,
    {
      "create_pod_tools": "create_pod_tools",
      "basic_setup": "basic_setup",
    },
  )
  graph.add_edge("create_pod_tools", "create_pod")
  graph.add_conditional_edges(
    "basic_setup",
    route_after_basic_setup,
    {
      "basic_setup_tools": "basic_setup_tools",
      "setup_intervention": "setup_intervention",
      "benchmark_loop": "benchmark_loop",
      "wrap_up": "wrap_up",
    },
  )
  graph.add_edge("basic_setup_tools", "basic_setup")
  graph.add_conditional_edges(
    "setup_intervention",
    route_after_setup_intervention,
    {
      "setup_intervention_tools": "setup_intervention_tools",
      "basic_setup": "basic_setup",
      "wrap_up": "wrap_up",
    },
  )
  graph.add_edge("setup_intervention_tools", "setup_intervention")
  graph.add_conditional_edges(
    "benchmark_loop",
    route_after_benchmark,
    {
      "benchmark_tools": "benchmark_tools",
      "wrap_up": "wrap_up",
    },
  )
  graph.add_edge("benchmark_tools", "benchmark_loop")
  graph.add_edge("wrap_up", END)
  return graph.compile(name="perferox_subagent")
