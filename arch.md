This is the rough architecture for the first version. The main thing is that we should not make the agent framework the architecture. Rig is useful, and probably good enough, but it should just be the thing that talks to models, calls tools, and lets us run an agent loop in Rust.

The actual Perferox core should be simple deterministic code. It should know what an experiment is, whether we have already tried something similar, how to run the command, how to store the result, and how to call out anomalies. The LLM can suggest stuff, but the boring Rust code should decide what gets written down.

The basic shape is probably:

- A Rust CLI/TUI.
- Rig for model providers, tool calling, structured extraction, embeddings, and the agent loop.
- SQLite as the local source of truth.
- JSONL traces for the main agent and subagents, so we can read what happened like Codex.
- Local runner first, then RunPod/Lambda runners later.

We should store three tables for now:

- `experiments`
- `runs`
- `anomalies`

For `experiments`, the important things are:

- agent no
- run no
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

- agent no
- run no
- started_at
- finished_at
- status
- trace_path
- stdout_path
- stderr_path

The trace should be a JSONL file, not a giant blob in SQLite. The TUI can render the trace live, but the file should be readable after the fact. This matters a lot because these runs will be weird and the agent will do a lot of small things. If we cannot read the trace cleanly, the project will be annoying to debug.

For `anomalies`, we need:

- agent no
- run no
- date
- summary
- command

The anomaly summary should be human readable. It should not just say "metric changed". It should say something like "TTFT p99 jumped on PD disaggregation with HiCache enabled, while output throughput stayed roughly flat", or "GLM-5.2 generated garbled output after cache hits with MTP enabled".

We should not store both command and normalized_command. Just store one canonical command string. If the agent wants to keep the messy original thing it wrote, that can live in the trace. The DB should hold the stable command we actually ran.

I think we do still want an id-ish thing, but it does not have to be a fake global id. Since run no is per agent, `(agent_no, run_no)` can be the natural key for now. SQLite has rowids anyway. If this gets annoying later, adding explicit ids is easy.

The experiment repeat check should have two layers:

- exact_hash, which catches the same exact experiment spec
- intent_key + embedding, which catches experiments that are basically testing the same thing

The intent_key should be a string, because I want it to be readable and searchable. Something like:

`SGLang CUDA radix cache long context throughput regression`

Then we embed that string and search previous experiment embeddings. Embeddings should not silently skip stuff. They should bring related experiments to the agent's attention. The agent can then decide if the new thing is a duplicate, related but still worth running, or actually new.

The TUI should be very basic at first. A few boxes where we put in target info and objective, then after it starts we mainly see:

- session status
- main agent trace
- subagent notes
- current experiment/run
- anomalies

It should feel more like a cockpit than a website. Ratatui is probably the right Rust lib for this, with crossterm underneath. The important part is not beautiful UI. The important part is that the trace is readable while the agent is running, and also replayable later from the JSONL file.

Rig should be used for:

- model providers
- tool calling
- structured extraction
- embeddings
- agent loop

Rig should not be used as our memory system or DB. SQLite and trace files are the memory system for now.

The first runner should just be local. It runs the command, captures stdout/stderr, records timings, samples GPU memory if possible, and writes the SGLang benchmark output into the trace/artifact folder. Later we add runners for RunPod or Lambda, but they should have the same shape as the local runner.

For SGLang, the default benchmark command should probably use:

`--output-details --cache-report`

This gives us the detailed errors, TTFT arrays, ITL arrays, generated texts, cache hit info, and the normal throughput/latency summaries. We can keep the raw benchmark JSON as an artifact and only lift the important fields into SQLite.

The main thing I want to avoid is making this too smart too early. The agent can be clever, but the storage and runner should be boring. If this works, it should be because we can run a lot of odd SGLang configs and reliably remember what happened, not because we invented a huge framework.
