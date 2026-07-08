"""LangGraph lifecycle and JSONL tracing for one benchmark subagent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from perferox.prompts import BENCHMARK_SYSTEM_PROMPT, SETUP_SYSTEM_PROMPT
from perferox.tools import runpodctl


class SubagentState(TypedDict, total=False):
  """Graph-safe state for one benchmark subagent."""

  agent_id: int
  messages: Annotated[list[AnyMessage], add_messages]
  summary: str

def trace_jsonable(value: Any) -> Any:
  """Convert trace payload objects JSON cannot encode."""
  if isinstance(value, Path):
    return str(value)
  if hasattr(value, "model_dump"):
    return value.model_dump()
  return repr(value)


def stream_with_trace(
  graph: Any,
  state: SubagentState,
  path: str | Path,
):
  """Stream graph updates while appending each public event to JSONL."""
  agent_id = int(state["agent_id"])
  trace_file = Path(path)
  trace_file.parent.mkdir(parents=True, exist_ok=True)
  for event in graph.stream(state, stream_mode="updates"):
    record = {
      "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
      "agent_id": agent_id,
      "kind": "graph_update",
      "payload": event,
    }
    with trace_file.open("a", encoding="utf-8") as file:
      file.write(json.dumps(record, separators=(",", ":"), default=trace_jsonable) + "\n")
    yield event


def _model_node(model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str):
  """Return a LangGraph node that invokes one chat model step."""
  bound_model = model.bind_tools(list(tools)) if tools else model

  def call_model(state: SubagentState):
    """Ask the model for the next setup or benchmark action."""
    messages = [SystemMessage(content=system_prompt), *state.get("messages", [])]
    response = bound_model.invoke(messages)
    return {"messages": [response]}

  return call_model


def _add_tool_phase(
  graph: StateGraph,
  model: BaseChatModel,
  name: str,
  tool_node: str,
  tools: Sequence[BaseTool],
  prompt: str,
  description: str,
) -> None:
  """Add one model phase and its matching tool-execution node."""
  graph.add_node(name, _model_node(model, tools, prompt), metadata={"description": description})
  graph.add_node(tool_node, ToolNode(tools, name=tool_node), metadata={"description": f"Run tools requested by {name}."})


def _create_pod_node(model: BaseChatModel, tool: BaseTool):
  """Return one node that loops on runpodctl until the model stops calling it."""
  bound_model = model.bind_tools([tool])
  tool_node = ToolNode((tool,))

  def create_pod(state: SubagentState):
    """Create a pod, wait for SSH details, and return the messages LangGraph should keep."""
    messages = [SystemMessage(content=SETUP_SYSTEM_PROMPT), *state.get("messages", [])]
    updates: list[AnyMessage] = []
    while True:
      response = bound_model.invoke(messages)
      messages.append(response)
      updates.append(response)
      if not getattr(response, "tool_calls", None):
        return {"messages": updates}
      tool_update = tool_node.invoke({"messages": messages})
      tool_messages = tool_update.get("messages", [])
      messages.extend(tool_messages)
      updates.extend(tool_messages)

  return create_pod


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


def _route_after_setup_intervention(state: SubagentState) -> str:
  """Route setup intervention back to setup or out to wrap-up."""
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "setup_intervention_tools"
  if "setup_failed" in _message_text(last_message).lower():
    return "wrap_up"
  return "basic_setup"


def _route_after_benchmark(state: SubagentState) -> str:
  """Keep cycling through benchmark tools until the model stops calling them."""
  last_message = state["messages"][-1]
  if getattr(last_message, "tool_calls", None):
    return "benchmark_tools"
  return "wrap_up"


def _wrap_up(state: SubagentState):
  """Persist the latest assistant text as the subagent summary."""
  summary = _message_text(state["messages"][-1])
  return {"summary": summary}


def build_subagent_graph(
  model: BaseChatModel,
  runpodctl_tool: BaseTool = runpodctl,
  setup_tools: Sequence[BaseTool] = (),
  benchmark_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
  """Compile the fixed subagent lifecycle graph."""
  graph = StateGraph(SubagentState)
  graph.add_node(
    "create_pod",
    _create_pod_node(model, runpodctl_tool),
    metadata={"description": "Create one temporary pod and wait for SSH details."},
  )
  for name, tool_node, tools, prompt, description in (
    ("basic_setup", "basic_setup_tools", setup_tools, SETUP_SYSTEM_PROMPT, "Prepare the pod to run SGLang serving benchmarks."),
    ("setup_intervention", "setup_intervention_tools", setup_tools, SETUP_SYSTEM_PROMPT, "Recover from setup_failed and retry setup if useful."),
    ("benchmark_loop", "benchmark_tools", benchmark_tools, BENCHMARK_SYSTEM_PROMPT, "Run bounded SGLang benchmark experiments for the goal."),
  ):
    _add_tool_phase(graph, model, name, tool_node, tools, prompt, description)
  graph.add_node(
    "wrap_up",
    _wrap_up,
    metadata={"description": "Save the final worker summary into graph state."},
  )

  graph.add_edge(START, "create_pod")
  graph.add_edge("create_pod", "basic_setup")
  graph.add_conditional_edges(
    "basic_setup",
    _route_after_basic_setup,
    {
      "basic_setup_tools": "basic_setup_tools",
      "setup_intervention": "setup_intervention",
      "benchmark_loop": "benchmark_loop",
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
    _route_after_benchmark,
    {
      "benchmark_tools": "benchmark_tools",
      "wrap_up": "wrap_up",
    },
  )
  graph.add_edge("benchmark_tools", "benchmark_loop")
  graph.add_edge("wrap_up", END)
  return graph.compile(name="perferox_subagent")
