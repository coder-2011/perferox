"""LangGraph lifecycle and JSONL tracing for one benchmark subagent."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
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
  messages: Annotated[list[AnyMessage], add_messages]
  summary: str

def trace_jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if hasattr(value, "model_dump"):
    return value.model_dump()
  return repr(value)


def stream_with_trace(
  graph: Any,
  state: SubagentState,
  path: str | Path,
  teardown: Callable[[], None],
  session_registry: SessionRegistry,
) -> Iterator[Any]:
  """Stream graph updates, write JSONL, call teardown, and close SSH."""
  agent_id = int(state["agent_id"])
  trace_file = Path(path)
  trace_file.parent.mkdir(parents=True, exist_ok=True)
  try:
    for event in graph.stream(state, stream_mode="updates"):
      record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_id": agent_id,
        "kind": "graph_update",
        "payload": event,
      }
      with trace_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, separators=(",", ":"), default=trace_jsonable) + "\n")
      yield event
  finally:
    try:
      teardown()
    finally:
      session_registry.close(f"agent-{agent_id}")


def _model_node(model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str) -> Callable[[SubagentState], dict[str, list[BaseMessage]]]:
  """Return a LangGraph node that invokes one chat model step."""
  bound_model = model.bind_tools(list(tools)) if tools else model

  def call_model(state: SubagentState) -> dict[str, list[BaseMessage]]:
    messages = [SystemMessage(content=system_prompt), *state.get("messages", [])]
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


def _route_after_basic_setup(state: SubagentState) -> str:
  """Choose whether setup needs tools, intervention, or benchmarking."""
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "basic_setup_tools"
  if "setup_failed" in _message_text(last_message).lower():
    return "setup_intervention"
  return "benchmark_loop"


def _route_after_create_pod(state: SubagentState) -> str:
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "create_pod_tools"
  return "basic_setup"


def _route_after_setup_intervention(state: SubagentState) -> str:
  """Route setup intervention back to setup or out to wrap-up."""
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "setup_intervention_tools"
  if "setup_failed" in _message_text(last_message).lower():
    return "wrap_up"
  return "basic_setup"


def _route_after_benchmark(state: SubagentState) -> str:
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "benchmark_tools"
  return "wrap_up"


def _logged_successes(state: SubagentState) -> int:
  """Count successful experiment-log tool outputs for cap routing."""
  return sum(
    getattr(message, "type", "") == "tool"
    and getattr(message, "name", "") == "log_experiment"
    and str(getattr(message, "content", "")).startswith("logged experiment")
    for message in state.get("messages", [])
  )


def _wrap_up(state: SubagentState) -> dict[str, str]:
  summary = _message_text(state["messages"][-1])
  return {"summary": summary}


def build_subagent_graph(
  model: BaseChatModel,
  agent_id: int,
  session_registry: SessionRegistry,
  db_path: str | Path,
  experiment_cap: int = 1,
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
    + f"\nHard cap: log at most {experiment_cap} successful benchmark experiment(s), then summarize."
  )
  remote_tool = remote_terminal(session_registry, session_id)
  create_tool_list.append(connect_remote_session(session_registry, session_id))
  setup_tool_list.append(remote_tool)
  benchmark_tool_list.append(remote_tool)
  benchmark_tool_list.append(sglang_bench_serving(session_registry, session_id, db_path, agent_id, experiment_cap, trace_ref))
  benchmark_tool_list.append(log_experiment_tool(db_path, agent_id))
  benchmark_tool_list.append(log_anomaly_tool(db_path, agent_id))

  def route_after_basic_setup(state: SubagentState) -> str:
    route = _route_after_basic_setup(state)
    if route == "benchmark_loop" and _logged_successes(state) >= experiment_cap:
      return "wrap_up"
    return route

  def route_after_benchmark(state: SubagentState) -> str:
    if _logged_successes(state) >= experiment_cap:
      return "wrap_up"
    return _route_after_benchmark(state)

  for name, tool_node, tools, prompt, description in (
    ("create_pod", "create_pod_tools", create_tool_list, CREATE_POD_SYSTEM_PROMPT, "Create one temporary pod and wait for SSH details."),
    ("basic_setup", "basic_setup_tools", setup_tool_list, SETUP_SYSTEM_PROMPT, "Prepare the pod to run SGLang serving benchmarks."),
    ("setup_intervention", "setup_intervention_tools", setup_tool_list, SETUP_SYSTEM_PROMPT, "Recover from setup_failed and retry setup if useful."),
    ("benchmark_loop", "benchmark_tools", benchmark_tool_list, benchmark_prompt, "Run bounded SGLang benchmark experiments for the goal."),
  ):
    graph.add_node(name, _model_node(model, tools, prompt), metadata={"description": description})
    graph.add_node(tool_node, ToolNode(tools, name=tool_node), metadata={"description": f"Run tools requested by {name}."})
  graph.add_node(
    "wrap_up",
    _wrap_up,
    metadata={"description": "Save the final worker summary into graph state."},
  )

  graph.add_edge(START, "create_pod")
  graph.add_conditional_edges(
    "create_pod",
    _route_after_create_pod,
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
    _route_after_setup_intervention,
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
