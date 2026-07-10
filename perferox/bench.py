"""SGLang benchmark command builders and parsers."""

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BENCH_TIMEOUT_S = 6 * 60 * 60.0
_CONSOLE_METRIC_LABELS = {
  "Request throughput (req/s)": "request_rps",
  "Input token throughput (tok/s)": "input_tps",
  "Output token throughput (tok/s)": "output_tps",
  "Median TTFT (ms)": "ttft_p50_ms",
  "P99 TTFT (ms)": "ttft_p99_ms",
  "Median TPOT (ms)": "tpot_p50_ms",
  "P99 TPOT (ms)": "tpot_p99_ms",
  "Cache hit rate": "cache_hit_rate",
  "Accept length": "accept_length",
}

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
  base_url: str | None = Field(None, description="Full server base URL; use instead of host/port when needed.", examples=["http://127.0.0.1:30000"])
  host: str | None = Field(None, description="Server host when base_url is not provided; SGLang normally uses 127.0.0.1.", examples=["127.0.0.1"])
  port: int | None = Field(None, ge=1, le=65535, description="Server port when base_url is not provided; SGLang normally uses 30000.", examples=[30000])
  ready_check_timeout_sec: int | None = Field(None, ge=0, description="Seconds to wait for readiness; 0 skips readiness polling.")
  dataset_name: BenchDataset = Field("sharegpt", description="Dataset generator or format; random is the simplest self-contained smoke test.")
  dataset_path: str | None = Field(None, description="Dataset file path; ShareGPT downloads its default data when omitted.")
  dataset_offset: int | None = Field(None, description="Rotate agentic-trace conversations by this many entries.")
  agentic_max_turns: int | None = Field(None, ge=1, description="Maximum turns per agentic-trace conversation.")
  speed_bench_category: Literal["low_entropy", "mixed", "high_entropy"] | None = Field(None, description="speed-bench category filter.")
  speed_bench_output_len: int | None = Field(None, ge=1, description="Fixed speed-bench output length.")
  model: str | None = Field(None, description="Exact served model name or path; normally pass this explicitly.", examples=["meta-llama/Llama-3.1-8B-Instruct"])
  served_model_name: str | None = Field(None, description="Model name exposed by the serving API.")
  tokenizer: str | None = Field(None, description="Tokenizer name or path.")
  num_prompts: int = Field(..., ge=1, description="Number of benchmark requests; required to bound the run.", examples=[100])
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
  output_file: str | None = Field(None, description="JSONL output path; SGLang chooses a name when details are enabled and this is omitted.")
  output_details: bool = Field(True, description="Write per-request inputs, outputs, errors, and latency details.")
  print_requests: bool = Field(False, description="Print each request while benchmarking.")
  disable_tqdm: bool = Field(False, description="Disable tqdm progress bar.")
  disable_stream: bool = Field(False, description="Use non-streaming requests where supported.")
  return_logprob: bool = Field(False, description="Request logprobs.")
  top_logprobs_num: int | None = Field(None, ge=0, description="Top logprobs per token.")
  token_ids_logprob: list[TokenId] | None = Field(None, description="Specific token IDs to probe for logprob.")
  logprob_start_len: int | None = Field(None, ge=-1, description="Input logprob start position; -1 disables input logprobs.")
  return_routed_experts: bool = Field(False, description="Return routed expert metadata.")
  cache_report: bool = Field(True, description="Collect cache hit statistics; useful for SGLang backends when the server enables cache reporting.")
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
  timeout_s: float = Field(BENCH_TIMEOUT_S, gt=0, le=BENCH_TIMEOUT_S, description="Host-side SSH timeout, capped at six hours; not a bench_serving flag.")

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
    if self.lora_request_distribution in ("distinct", "skewed") and len(self.lora_name or ()) <= 1:
      requirement = "lora_name" if not self.lora_name else "more than one lora_name"
      raise ValueError(f"distinct/skewed LoRA distribution requires {requirement}")
    return self


class BenchmarkRunArgs(BaseModel):
  """Expose a compact common-case benchmark schema with validated advanced flags."""

  model_config = ConfigDict(extra="forbid")

  intent_key: str = Field(..., min_length=3, description="Concise semantic purpose that distinguishes this experiment.", examples=["random 1k input latency at concurrency 32"])
  hardware_config: str = Field(..., min_length=2, description="Observed GPU or machine type, not the resource ID.", examples=["NVIDIA A100-SXM4-80GB"])
  server_config: str = Field(..., min_length=2, description="Exact server launch configuration that can affect results.", examples=["python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --tp 1"])
  num_prompts: int = Field(..., ge=1, description="Number of requests; required to bound the run.", examples=[100])
  model: str | None = Field(None, description="Exact served model name or path.")
  backend: BenchBackend = Field("sglang", description="Serving backend/API shape.")
  base_url: str | None = Field(None, description="Full server URL; use instead of host/port when needed.")
  host: str | None = Field(None, description="Server host, normally 127.0.0.1.")
  port: int | None = Field(None, ge=1, le=65535, description="Server port, normally 30000.")
  dataset_name: BenchDataset = Field("sharegpt", description="Dataset generator or format; random is self-contained.")
  dataset_path: str | None = Field(None, description="Dataset file path when required.")
  random_input_len: int | None = Field(None, ge=1, description="Random dataset input tokens.")
  random_output_len: int | None = Field(None, ge=0, description="Random dataset output tokens.")
  request_rate: float | None = Field(None, gt=0, description="Requests per second; omit for all-at-once.")
  max_concurrency: int | None = Field(None, ge=0, description="Maximum in-flight requests.")
  seed: int | None = Field(None, ge=0, le=4294967295)
  output_details: bool = Field(True, description="Write per-request details.")
  cache_report: bool = Field(True, description="Collect cache hit statistics.")
  timeout_s: float = Field(BENCH_TIMEOUT_S, gt=0, le=BENCH_TIMEOUT_S, description="Host SSH timeout, capped at six hours.")
  advanced_args: dict[str, Any] = Field(default_factory=dict, description="Rare BenchServingArgs fields only; names and values are still strictly validated.")


class ExperimentMetrics(BaseModel):
  """Expose the normalized metric keys accepted by SQLite to the model."""

  model_config = ConfigDict(extra="forbid")

  request_rps: float | None = Field(None, ge=0)
  input_tps: float | None = Field(None, ge=0)
  output_tps: float | None = Field(None, ge=0)
  ttft_p50_ms: float | None = Field(None, ge=0)
  ttft_p99_ms: float | None = Field(None, ge=0)
  tpot_p50_ms: float | None = Field(None, ge=0)
  tpot_p99_ms: float | None = Field(None, ge=0)
  error_rate: float | None = Field(None, ge=0, le=1)
  cache_hit_rate: float | None = Field(None, ge=0, le=1)
  peak_gpu_mem_gb: float | None = Field(None, ge=0)
  startup_s: float | None = Field(None, ge=0)
  warmup_s: float | None = Field(None, ge=0)
  accept_length: float | None = Field(None, ge=0)
  correctness_score: float | None = Field(None, ge=0)


def bench_serving_argv(args: BenchServingArgs) -> list[str]:
  """Build the SGLang serving benchmark argv from typed fields."""
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
    argv.extend(map(str, value if isinstance(value, list) else (value,)))
  return argv


def parse_bench_serving_metrics(output: str, expected_requests: int | None = None) -> dict[str, float]:
  """Extract Perferox experiment metrics from SGLang benchmark output."""
  metrics = {}
  for raw_line in output.splitlines():
    label, separator, raw_value = raw_line.partition(":")
    if not separator:
      continue
    label = label.strip()
    metric_name = _CONSOLE_METRIC_LABELS.get(label)
    if metric_name is None and (label != "Successful requests" or not expected_requests):
      continue
    raw_value = raw_value.strip()
    if not raw_value:
      continue
    raw_number = raw_value.split(maxsplit=1)[0].rstrip("%")
    try:
      value = float(raw_number)
    except ValueError:
      continue
    if metric_name is None:
      metrics["error_rate"] = max(expected_requests - value, 0.0) / expected_requests
      continue
    metrics[metric_name] = value / 100.0 if metric_name == "cache_hit_rate" else value
  return metrics
