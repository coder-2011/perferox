# ruff: noqa: BLE001

import json
import os
import shlex
import signal
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from perferox import db
from perferox.remote import RemoteSession, SessionRegistry

DEFAULT_TIMEOUT_S = 30.0
BENCH_TIMEOUT_S = 6 * 60 * 60.0
MAX_OUTPUT_CHARS = 10000

BenchBackend = Literal[
  "sglang",
  "sglang-native",
  "sglang-oai",
  "sglang-oai-chat",
  "sglang-embedding",
  "vllm",
  "vllm-chat",
  "lmdeploy",
  "lmdeploy-chat",
  "trt",
  "gserver",
  "truss",
]
BenchDataset = Literal[
  "agentic-trace",
  "autobench",
  "sharegpt",
  "custom",
  "openai",
  "random",
  "random-ids",
  "generated-shared-prefix",
  "mmmu",
  "image",
  "mooncake",
  "longbench_v2",
  "speed-bench",
]
ProfileActivity = Literal["CPU", "GPU", "CUDA_PROFILER", "XPU", "MEM"]
TokenId = Annotated[int, Field(ge=0)]

_EMBEDDING_UNSUPPORTED_DATASETS = {"image", "mmmu", "mooncake"}


class BenchServingArgs(BaseModel):
  """Typed inputs for SGLang's serving benchmark CLI."""

  model_config = ConfigDict(extra="forbid")

  backend: BenchBackend = Field("sglang", description="Serving backend/API shape to benchmark.")
  base_url: str | None = Field(None, description="Full server base URL; use instead of host/port when needed.")
  host: str | None = Field(None, description="Server host when base_url is not provided.")
  port: int | None = Field(None, ge=1, le=65535, description="Server port when base_url is not provided.")
  ready_check_timeout_sec: int | None = Field(None, ge=0, description="Seconds to wait for readiness; 0 skips readiness polling.")
  dataset_name: BenchDataset = Field("sharegpt", description="Dataset generator or dataset format.")
  dataset_path: str | None = Field(None, description="Path to dataset file when the selected dataset needs one.")
  dataset_offset: int | None = Field(None, description="Rotate agentic-trace conversations by this many entries.")
  agentic_max_turns: int | None = Field(None, ge=1, description="Maximum turns per agentic-trace conversation.")
  speed_bench_category: Literal["low_entropy", "mixed", "high_entropy"] | None = Field(None, description="speed-bench category filter.")
  speed_bench_output_len: int | None = Field(None, ge=1, description="Fixed speed-bench output length.")
  model: str | None = Field(None, description="Model name or path.")
  served_model_name: str | None = Field(None, description="Model name exposed by the serving API.")
  tokenizer: str | None = Field(None, description="Tokenizer name or path.")
  num_prompts: int | None = Field(None, ge=1, description="Number of benchmark requests.")
  sharegpt_output_len: int | None = Field(None, ge=4, description="Override ShareGPT output length.")
  sharegpt_context_len: int | None = Field(None, ge=1, description="Drop ShareGPT requests longer than this context length.")
  random_input_len: int | None = Field(None, ge=1, description="Random/image dataset input tokens per request.")
  random_output_len: int | None = Field(None, ge=0, description="Random/image dataset output tokens per request.")
  random_range_ratio: float | None = Field(None, ge=0, le=1, description="Input/output length sampling range ratio.")
  image_count: int | None = Field(None, ge=1, description="Images per request for image dataset.")
  image_resolution: str | None = Field(
    None,
    pattern=r"^(4k|1080p|720p|360p|[1-9]\d*x[1-9]\d*)$",
    description="Image dataset resolution preset or HxW string.",
  )
  random_image_count: bool = Field(False, description="Randomize image count for image dataset.")
  image_format: Literal["jpeg", "png"] | None = Field(None, description="Image format for image dataset.")
  image_content: Literal["random", "blank"] | None = Field(None, description="Image content type for image dataset.")
  request_rate: float | None = Field(None, gt=0, description="Requests per second; omit for bench_serving's all-at-once default.")
  use_trace_timestamps: bool = Field(False, description="Replay mooncake trace timestamps.")
  max_concurrency: int | None = Field(None, ge=0, description="Maximum concurrent in-flight requests; 0 behaves like unset.")
  output_file: str | None = Field(None, description="JSONL output file path.")
  output_details: bool = Field(False, description="Write per-request benchmark details.")
  print_requests: bool = Field(False, description="Print each request while benchmarking.")
  disable_tqdm: bool = Field(False, description="Disable tqdm progress bar.")
  disable_stream: bool = Field(False, description="Use non-streaming requests where supported.")
  return_logprob: bool = Field(False, description="Request logprobs.")
  top_logprobs_num: int | None = Field(None, ge=0, description="Top logprobs per token.")
  token_ids_logprob: list[TokenId] | None = Field(None, description="Specific token IDs to probe for logprob.")
  logprob_start_len: int | None = Field(None, ge=-1, description="Input logprob start position; -1 disables input logprobs.")
  return_routed_experts: bool = Field(False, description="Return routed expert metadata.")
  cache_report: bool = Field(False, description="Collect cache hit statistics.")
  seed: int | None = Field(None, ge=0, le=4294967295, description="Random seed accepted by numpy.")
  disable_ignore_eos: bool = Field(False, description="Respect EOS instead of forcing fixed output length.")
  temperature: float | None = Field(None, ge=0, description="Sampling temperature.")
  top_p: float | None = Field(None, gt=0, le=1, description="Nucleus sampling top-p.")
  extra_request_body: dict[str, Any] | None = Field(None, description="Extra JSON request body merged into each request.")
  apply_chat_template: bool = Field(False, description="Apply tokenizer chat template.")
  profile: bool = Field(False, description="Use Torch profiler endpoints.")
  plot_throughput: bool = Field(False, description="Plot throughput over time.")
  profile_activities: list[ProfileActivity] | None = Field(None, description="Torch profiler activities.")
  profile_start_step: int | None = Field(None, ge=0, description="Start profiler after this many steps.")
  profile_steps: int | None = Field(None, ge=1, description="Number of profiler steps.")
  profile_num_steps: int | None = Field(None, ge=1, description="Profiler num_steps body field.")
  profile_by_stage: bool = Field(False, description="Profile by serving stage.")
  profile_stages: list[str] | None = Field(None, description="Serving stages to profile.")
  profile_output_dir: str | None = Field(None, description="Profiler output directory.")
  profile_prefix: str | None = Field(None, description="Profiler trace filename prefix.")
  lora_name: list[str] | None = Field(None, description="LoRA adapter names.")
  lora_request_distribution: Literal["uniform", "distinct", "skewed"] | None = Field(None, description="LoRA request sampling distribution.")
  lora_zipf_alpha: float | None = Field(None, gt=1, description="Zipf alpha for skewed LoRA distribution; omit for bench_serving default 1.5.")
  prompt_suffix: str | None = Field(None, description="Suffix appended to prompts.")
  pd_separated: bool = Field(False, description="Benchmark prefill/decode disaggregated serving.")
  profile_prefill_url: list[str] | None = Field(None, description="Prefill worker profiling URLs.")
  profile_decode_url: list[str] | None = Field(None, description="Decode worker profiling URLs.")
  flush_cache: bool = Field(False, description="Flush server cache before benchmarking.")
  warmup_requests: int | None = Field(None, ge=0, description="Warmup requests before timed benchmark.")
  tokenize_prompt: bool = Field(False, description="Send token IDs instead of prompt text.")
  gsp_num_groups: int | None = Field(None, ge=1, description="Generated-shared-prefix group count.")
  gsp_prompts_per_group: int | None = Field(None, ge=1, description="Generated-shared-prefix prompts per group.")
  gsp_system_prompt_len: int | None = Field(None, ge=0, description="Generated-shared-prefix system prompt tokens.")
  gsp_question_len: int | None = Field(None, ge=0, description="Generated-shared-prefix question tokens.")
  gsp_output_len: int | None = Field(None, ge=0, description="Generated-shared-prefix output tokens.")
  gsp_range_ratio: float | None = Field(None, ge=0, le=1, description="Generated-shared-prefix length range ratio.")
  gsp_fast_prepare: bool = Field(False, description="Skip slow generated-shared-prefix preparation stats.")
  gsp_send_routing_key: bool = Field(False, description="Send routing key header for shared-prefix requests.")
  gsp_num_turns: int | None = Field(None, ge=1, description="Generated-shared-prefix turns per prompt.")
  gsp_ordered: bool = Field(False, description="Keep generated-shared-prefix requests ordered.")
  gsp_group_distribution: Literal["uniform", "zipf"] | None = Field(None, description="Generated-shared-prefix group sampling distribution.")
  gsp_zipf_alpha: float | None = Field(None, gt=0, allow_inf_nan=False, description="Zipf alpha for generated-shared-prefix group distribution.")
  mooncake_slowdown_factor: float | None = Field(None, gt=0, description="Slowdown factor for mooncake trace replay.")
  mooncake_num_rounds: int | None = Field(None, ge=1, description="Conversation rounds for mooncake dataset.")
  mooncake_workload: Literal["mooncake", "conversation", "synthetic", "toolagent"] | None = Field(None, description="Mooncake workload type.")
  fake_prefill: bool = Field(False, description="Use fake prefill mode for decode-only benchmarking.")
  tag: str | None = Field(None, description="Tag written to benchmark output.")
  header: dict[str, str] | None = Field(None, description="Custom HTTP headers as key/value pairs.")
  timeout_s: float | None = Field(BENCH_TIMEOUT_S, gt=0, description="Host-side SSH command timeout; not a bench_serving flag.")

  @model_validator(mode="after")
  def check_serving_constraints(self) -> Self:
    """Mirror bench_serving's parse-time and early runtime argument checks."""
    if self.gsp_group_distribution == "zipf" and self.gsp_zipf_alpha is None:
      raise ValueError("gsp_group_distribution='zipf' requires gsp_zipf_alpha")
    if self.gsp_zipf_alpha is not None and self.gsp_group_distribution != "zipf":
      raise ValueError("gsp_zipf_alpha is only valid when gsp_group_distribution='zipf'")
    if self.backend in ("trt", "truss") and self.model is None:
      raise ValueError(f"backend='{self.backend}' requires model")
    if self.profile_prefill_url and self.profile_decode_url:
      raise ValueError("profile_prefill_url and profile_decode_url are mutually exclusive")
    if self.print_requests and self.backend != "sglang-oai-chat":
      raise ValueError("print_requests is only supported with backend='sglang-oai-chat'")
    if self.tokenize_prompt and self.backend != "sglang":
      raise ValueError("tokenize_prompt is only supported with backend='sglang'")
    if self.tokenize_prompt and self.dataset_name in ("image", "mmmu"):
      raise ValueError("tokenize_prompt is not compatible with image or mmmu datasets")
    if self.backend == "sglang-embedding" and self.dataset_name in _EMBEDDING_UNSUPPORTED_DATASETS:
      raise ValueError(f"{self.dataset_name} is unsupported for sglang-embedding")
    if self.lora_request_distribution in ("distinct", "skewed") and not self.lora_name:
      raise ValueError("distinct/skewed LoRA distribution requires lora_name")
    if self.lora_request_distribution in ("distinct", "skewed") and len(self.lora_name or []) <= 1:
      raise ValueError("distinct/skewed LoRA distribution requires more than one lora_name")
    return self


@tool(
  "local_terminal",
  description="Run one shell command on the local host. Use for local files and local setup; directory changes do not persist.",
)
def local_terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
  return _run_local(command, timeout_s)


def connect_remote_session(registry: SessionRegistry, session_id: str) -> BaseTool:
  @tool(
    "connect_remote_session",
    description="Open the persistent SSH session after local runpodctl returns host, user, and port.",
  )
  def connect(host: str, user: str = "root", port: int = 22, timeout_s: float = 30.0) -> str:
    registry.close(session_id)
    try:
      registry.add(RemoteSession.connect(session_id, host, user, port, timeout_s))
    except Exception as exc:
      return f"remote session connect failed: {type(exc).__name__}: {exc}"
    return f"connected remote session {session_id} to {user}@{host}:{port}"

  return connect


def remote_terminal(registry: SessionRegistry, session_id: str) -> BaseTool:
  @tool(
    "remote_terminal",
    description="Run one shell command on the connected remote SSH machine.",
  )
  def terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
    try:
      session = registry.get(session_id)
    except KeyError:
      return _format_result(None, "", f"remote session not connected: {session_id}")
    return _run_remote(session, command, timeout_s)

  return terminal


def sglang_bench_serving(
  registry: SessionRegistry,
  session_id: str,
  db_path: str | Path,
  agent_id: int,
  experiment_cap: int,
  trace_ref: str = "",
) -> BaseTool:
  @tool(
    "sglang_bench_serving",
    args_schema=BenchServingArgs,
    description="Run one structured SGLang serving benchmark on the connected remote SSH machine and return its host-assigned run_id.",
  )
  def run(**kwargs: Any) -> str:
    """Run one benchmark command after the host assigns its run id."""
    try:
      args = BenchServingArgs(**kwargs)
      command = " ".join(shlex.quote(part) for part in _bench_serving_argv(args))
    except Exception as exc:
      return f"invalid bench_serving args: {type(exc).__name__}: {exc}"
    session = registry.get(session_id)
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.start_benchmark_run(conn, agent_id=agent_id, command=command, experiment_cap=experiment_cap, trace_ref=trace_ref)
    except Exception as exc:
      return f"benchmark not started: {type(exc).__name__}: {exc}"
    output = _run_remote(session, command, args.timeout_s)
    if not output.startswith("exit_code=0\n"):
      # Failed started runs still count, so mark them in SQLite.
      with closing(db.connect(db_path)) as conn:
        db.mark_run_failed(conn, agent_id=agent_id, run_id=run_id, error=output)
    return f"run_id={run_id}\n{output}"

  return run


def log_experiment_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  @tool(
    "log_experiment",
    description="Locally mark a successful benchmark run and save its metrics to SQLite.",
  )
  def log_experiment(intent_key: str, metrics: dict[str, float | int | None] | None = None) -> str:
    try:
      with closing(db.connect(db_path)) as conn:
        run_id = db.log_experiment(conn, agent_id=agent_id, intent_key=intent_key, metrics=metrics)
    except Exception as exc:
      return f"log_experiment failed: {type(exc).__name__}: {exc}"
    return f"logged experiment agent_id={agent_id} run_id={run_id}"

  return log_experiment


def log_anomaly_tool(db_path: str | Path, agent_id: int) -> BaseTool:
  @tool(
    "log_anomaly",
    description="Locally save a human-readable anomaly for a benchmark run to SQLite.",
  )
  def log_anomaly(run_id: int, summary: str) -> str:
    try:
      with closing(db.connect(db_path)) as conn:
        anomaly_id = db.log_anomaly(conn, agent_id=agent_id, run_id=run_id, summary=summary)
    except Exception as exc:
      return f"log_anomaly failed: {type(exc).__name__}: {exc}"
    return f"logged anomaly anomaly_id={anomaly_id} agent_id={agent_id} run_id={run_id}"

  return log_anomaly


def _bench_serving_argv(args: BenchServingArgs) -> list[str]:
  """Build the serving benchmark command from typed fields."""
  data = args.model_dump(exclude={"timeout_s", "extra_request_body", "header"}, exclude_none=True)
  data["extra_request_body"] = json.dumps(args.extra_request_body, separators=(",", ":")) if args.extra_request_body else None
  data["header"] = [f"{key}={value}" for key, value in (args.header or {}).items()]
  argv = ["python", "-m", "sglang.benchmark.serving"]
  for name, value in data.items():
    if value is False or value is None or value == []:
      continue
    argv.append(f"--{name.replace('_', '-')}")
    if value is True:
      continue
    argv.extend(str(item) for item in (value if isinstance(value, list) else [value]))
  return argv


def _run_local(command: str, timeout_s: float | None) -> str:
  """Run a local command, killing the process group on timeout."""
  try:
    process = subprocess.Popen(
      ["bash", "-lc", command],
      text=True,
      encoding="utf-8",
      errors="replace",
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      start_new_session=os.name == "posix",
    )
    stdout, stderr = process.communicate(timeout=timeout_s)
    return _format_result(process.returncode, stdout, stderr)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
    stdout, stderr = process.communicate()
    return _format_result(None, stdout, f"{stderr}\ntimed out after {timeout_s}s")
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")


def _run_remote(session: RemoteSession, command: str, timeout_s: float | None) -> str:
  """Run a command through SSH."""
  try:
    result = session.run(f"bash -lc {shlex.quote(command)}", timeout_s=timeout_s)
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")
  return _format_result(result.exit_status, result.stdout, result.stderr)


def _format_result(exit_status: int | None, stdout: str, stderr: str) -> str:
  """Format command output for a tool message."""
  output = "\n".join(part for part in (stdout, stderr) if part)
  if len(output) > MAX_OUTPUT_CHARS:
    keep = MAX_OUTPUT_CHARS // 2
    skipped = len(output) - MAX_OUTPUT_CHARS
    output = f"{output[:keep]}\n\n... {skipped} chars elided ...\n\n{output[-keep:]}"
  return f"exit_code={exit_status}\n{output}"
