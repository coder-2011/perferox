"""LangGraph lifecycle and JSONL tracing for one benchmark subagent."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
  log_anomaly_tool,
  log_experiment_tool,
  remote_terminal,
  sglang_bench_serving,
)

MAX_WORKER_MESSAGES = 24
MAX_CREATE_MODEL_CALLS = 8
MAX_SETUP_MODEL_CALLS = 12


def _merge_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
  """Merge LangGraph messages while retaining one protocol-valid recent window."""
  recent = list(add_messages(left, right)[-MAX_WORKER_MESSAGES:])
  while recent and recent[0].type == "tool":
    recent.pop(0)
  return recent


class SubagentState(TypedDict, total=False):
  """Graph-safe state for one benchmark subagent."""

  agent_id: int
  objective: str
  messages: Annotated[list[AnyMessage], _merge_messages]
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
  state: Mapping[str, Any] | None,
  path: str | Path,
  *,
  config: Mapping[str, Any] | None = None,
  agent_id: int | None = None,
) -> Iterator[Any]:
  """Stream graph updates and append them to one JSONL trace."""
  if state is not None:
    agent_id = state.get("agent_id", agent_id)
  trace_file = Path(path)
  trace_file.parent.mkdir(parents=True, exist_ok=True)
  with trace_file.open("a", encoding="utf-8", buffering=1) as file:
    try:
      stream_options = {"stream_mode": "updates"}
      if config is not None:
        stream_options.update(config=config, durability="sync")
      for event in graph.stream(state, **stream_options):
        record = {
          "ts": datetime.now(UTC).isoformat(timespec="seconds"),
          "agent_id": agent_id,
          "kind": "graph_update",
          "payload": event,
        }
        file.write(json.dumps(record, separators=(",", ":"), default=trace_jsonable) + "\n")
        yield event
    except Exception as exc:
      record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_id": agent_id,
        "kind": "graph_error",
        "payload": f"{type(exc).__name__}: {exc}",
      }
      file.write(json.dumps(record, separators=(",", ":")) + "\n")
      raise


def _model_node(
  model: BaseChatModel,
  tools: Sequence[BaseTool],
  system_prompt: str,
  model_call_cap: int,
  exhausted_message: str,
) -> Callable[[SubagentState], dict[str, list[BaseMessage]]]:
  """Return a model node with a host-owned phase turn cap."""
  # Native web search completes inside the model call; ToolNode handles local tools.
  bound_model = model.bind_tools([*tools, WEB_SEARCH_TOOL], parallel_tool_calls=False)
  model_calls = 0

  def call_model(state: SubagentState) -> dict[str, list[BaseMessage]]:
    """Invoke the model with the subagent goal in the system prompt."""
    nonlocal model_calls
    if model_calls >= model_call_cap:
      return {"messages": [AIMessage(content=exhausted_message)]}
    model_calls += 1
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state.get("objective", "(none)")), *state.get("messages", [])]
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


def _cancel_tool_calls(state: SubagentState) -> dict[str, list[ToolMessage]]:
  """Resolve every emitted call with a protocol-valid soft-stop result."""
  calls = getattr(state["messages"][-1], "tool_calls", ())
  messages = [ToolMessage(content="stop requested; tool call cancelled", tool_call_id=call["id"], name=call["name"]) for call in calls]
  return {"messages": messages}


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
  create_pod_tools: Sequence[BaseTool] = (),
  setup_tools: Sequence[BaseTool] = (),
  benchmark_tools: Sequence[BaseTool] = (),
  checkpointer: Any = None,
) -> CompiledStateGraph:
  """Compile the fixed subagent lifecycle graph."""
  session_id = f"agent-{agent_id}"
  target_prompt = f"\n\nTarget repository:\n{repository}\n\nTarget commit:\n{commit}"
  graph = StateGraph(SubagentState)
  remote_tool = remote_terminal(session_registry, session_id)
  create_pod_tools = [*create_pod_tools, connect_remote_session(session_registry, session_id)]
  setup_tools = [*setup_tools, remote_tool]
  intervention_tools = [*create_pod_tools, *setup_tools]
  benchmark_tools = [
    *benchmark_tools,
    remote_tool,
    sglang_bench_serving(
      session_registry,
      session_id,
      db_path,
      agent_id,
      repository,
      commit,
      trace_ref,
      attempt_cap,
    ),
    log_experiment_tool(db_path, agent_id),
    log_anomaly_tool(db_path, agent_id),
  ]
  with db.open_db(db_path) as conn:
    db.init_db(conn)

  def stop_requested() -> bool:
    """Read the host-owned stop flag."""
    with db.open_db(db_path, readonly=True) as conn:
      return db.stop_requested(conn, agent_id=agent_id)

  def target_ready() -> bool:
    """Verify the live checkout resolves to the delegated immutable commit."""
    try:
      session = session_registry.get(session_id)
      expected = shlex.quote(f"{commit}^{{commit}}")
      command = f"test \"$(git -C /workspace/target rev-parse HEAD)\" = \"$(git -C /workspace/target rev-parse {expected})\""
      return session.run(command, timeout_s=30).exit_status == 0
    except (KeyError, OSError, RuntimeError):
      return False

  def route_after_create_pod(state: SubagentState) -> Literal["create_pod_tools", "cancel_tools", "basic_setup", "setup_intervention", "wrap_up"]:
    """Execute emitted calls, then advance only with a live SSH session."""
    last_message = state["messages"][-1]
    stopped = stop_requested()
    if getattr(last_message, "tool_calls", None):
      return "cancel_tools" if stopped else "create_pod_tools"
    if stopped:
      return "wrap_up"
    return "basic_setup" if session_registry.connected(session_id) else "setup_intervention"

  def route_after_basic_setup(state: SubagentState) -> Literal["basic_setup_tools", "cancel_tools", "basic_setup", "setup_intervention", "benchmark_loop", "wrap_up"]:
    """Choose setup tools, intervention, benchmark, or wrap-up."""
    last_message = state["messages"][-1]
    stopped = stop_requested()
    if getattr(last_message, "tool_calls", None):
      return "cancel_tools" if stopped else "basic_setup_tools"
    text = _message_text(last_message).lower()
    if stopped:
      return "wrap_up"
    if "setup_ready" in text:
      return "benchmark_loop" if target_ready() else "setup_intervention"
    if "setup_failed" in text:
      return "setup_intervention"
    return "basic_setup"

  def route_after_setup_intervention(state: SubagentState) -> Literal["setup_intervention_tools", "cancel_tools", "basic_setup", "benchmark_loop", "wrap_up"]:
    """Route setup intervention back to setup or out to wrap-up."""
    last_message = state["messages"][-1]
    stopped = stop_requested()
    if getattr(last_message, "tool_calls", None):
      return "cancel_tools" if stopped else "setup_intervention_tools"
    text = _message_text(last_message).lower()
    if stopped or "setup_failed" in text:
      return "wrap_up"
    if "setup_ready" in text and target_ready():
      return "benchmark_loop"
    return "basic_setup"

  def route_after_benchmark(state: SubagentState) -> Literal["benchmark_tools", "wrap_up"]:
    """Keep benchmarking until the started-attempt cap is reached."""
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
    with db.open_db(db_path, readonly=True) as conn:
      started_attempts = conn.execute("SELECT COUNT(*) FROM runs WHERE agent_id = ?", (agent_id,)).fetchone()[0]
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
    with db.open_db(db_path) as conn, conn:
      db.notify_main(conn, agent_id=agent_id, run_id=None, kind="subagent_summary", table_name="subagent_summary", row=row)
    return {"summary": summary, "messages": [response]}

  for name, tool_node, tools, prompt, model_call_cap, exhausted_message in (
    ("create_pod", "create_pod_tools", create_pod_tools, create_pod_prompt, MAX_CREATE_MODEL_CALLS, "provision_failed: host model-turn cap reached"),
    ("basic_setup", "basic_setup_tools", setup_tools, SETUP_SYSTEM_PROMPT, MAX_SETUP_MODEL_CALLS, "setup_failed: host model-turn cap reached"),
    ("setup_intervention", "setup_intervention_tools", intervention_tools, SETUP_SYSTEM_PROMPT, MAX_SETUP_MODEL_CALLS, "setup_failed: host intervention-turn cap reached"),
    ("benchmark_loop", "benchmark_tools", benchmark_tools, BENCHMARK_SYSTEM_PROMPT, attempt_cap * 3 + 4, "benchmark phase host model-turn cap reached"),
  ):
    phase_prompt = f"{prompt}{target_prompt}\n\nHard cap: start at most {attempt_cap} benchmark attempt(s), then summarize."
    graph.add_node(name, _model_node(model, tools, phase_prompt, model_call_cap, exhausted_message))
    graph.add_node(tool_node, ToolNode(tools, name=tool_node, handle_tool_errors=True))
  graph.add_node("wrap_up", wrap_up)
  graph.add_node("cancel_tools", _cancel_tool_calls)

  graph.add_edge(START, "create_pod")
  graph.add_conditional_edges("create_pod", route_after_create_pod)
  graph.add_edge("create_pod_tools", "create_pod")
  graph.add_conditional_edges("basic_setup", route_after_basic_setup)
  graph.add_edge("basic_setup_tools", "basic_setup")
  graph.add_conditional_edges("setup_intervention", route_after_setup_intervention)
  graph.add_edge("setup_intervention_tools", "setup_intervention")
  graph.add_conditional_edges("benchmark_loop", route_after_benchmark)
  graph.add_edge("benchmark_tools", "benchmark_loop")
  graph.add_edge("cancel_tools", "wrap_up")
  graph.add_edge("wrap_up", END)
  return graph.compile(name="perferox_subagent", checkpointer=checkpointer)
