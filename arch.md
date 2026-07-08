This is the rough architecture for the first version.

The basic shape is probably:

- A Rust CLI/TUI using Ratatui+crossterm
- Rig for agent stuff and tool calling
- SQLite as the local source of truth.
- JSONL traces for the main agent and subagents, so we can read what happened like Codex. I hope rig can handle this.


We should store three tables for now:

- `experiments`
- `runs`
- `anomalies`

For `experiments`, the important things are:

- agent #
- run #
- date
- intent_key
- intent_embedding
- command
- exact_hash
- request_rps
- input_tps
- output_tps
- ttft_p50_ms
- ttft_p99_ms
- tpot_p50_ms
- tpot_p99_ms
- error_rate
- cache_hit_rate
- peak_gpu_mem_gb
- startup_s
- warmup_s
- accept_length
- correctness_score

Some of these come from SGLang directly and some do not. SGLang's serving benchmark records request throughput, input throughput, output throughput, TTFT, TPOT, accept length, and cache hit rate if we run it with cache reporting. Error rate can be derived from the detailed errors. Peak GPU memory, startup time, warmup time, and correctness score are things Perferox should record itself.

For `runs`, we probably only need:

- agent #
- run #
- started_at
- finished_at
- status
- trace_path
- stdout_path
- stderr_path

The trace should be a JSONL file, not a giant blob in SQLite. The TUI can render the trace live, but the file should be readable after the fact. This matters a lot because these runs will be weird and the agent will do a lot of small things. If we cannot read the trace cleanly, the project will be annoying to debug.

For `anomalies`, we need:

- agent #
- run #
- date
- summary
- command

The anomaly summary should be human readable. ex. "GLM-5.2 generated garbled output after cache hits with MTP enabled, when we use DSpark w/ batch=1 and triton backend".

The experiment repeat check should have two layers:

- exact_hash, which catches the same exact experiment spec
- intent_key + embedding, which catches experiments that are basically testing the same thing

The intent_key should be a string, because I want it to be readable and searchable. Something like:

`SGLang CUDA radix cache long context throughput regression`

The TUI should be very basic at first. A few boxes where we put in target info and objective, then after it starts we mainly see main agent trace, maybe anomalies, and basic status indicators, like run #'s, what subagents are running, on what chips. Simple tui tho.

For SGLang, the default benchmark command should probably use:

`--output-details --cache-report`
