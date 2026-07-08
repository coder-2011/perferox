This is the rough architecture for the first version:

The basic shape is probably:

Textual for TUI
Langchain+graph for agent stuff (the initial version is meant to be pretty quick and dirty, performance of this agent isnt too important)
SQLite as the local source of truth.
JSONL traces for the main agent and subagents, so we can read what happened like Codex. I hope rig can handle this.
We should store three tables for now:

experiments
runs
anomalies
For experiments, the important things are:

agent #
run #
gpu
date
intent_key
intent_embedding
command
exact_hash
request_rps
input_tps
output_tps
ttft_p50_ms
ttft_p99_ms
tpot_p50_ms
tpot_p99_ms
error_rate
cache_hit_rate
peak_gpu_mem_gb
startup_s
warmup_s
accept_length
correctness_score
Some of these come from SGLang directly and some do not. SGLang's serving benchmark records request throughput, input throughput, output throughput, TTFT, TPOT, accept length, and cache hit rate if we run it with cache reporting. Error rate can be derived from the detailed errors. Peak GPU memory, startup time, warmup time, and correctness score are things Perferox should record itself.

For runs, we probably only need:

agent #
run #
gpu
started_at
finished_at
trace_path
For anomalies, we need:

agent #
run #
gpu
date
summary
command
The anomaly summary should be human readable. ex. "GLM-5.2 generated garbled output after cache hits with MTP enabled, when we use DSpark w/ batch=1 and triton backend".

The experiment repeat check should have two layers:

exact_hash, which catches the same exact experiment spec
intent_key + embedding, which catches experiments that are basically testing the same thing
The intent_key should be a string, because I want it to be readable and searchable. Something like:

SGLang CUDA radix cache long context throughput regression

The TUI should be very basic at first. A few boxes where we put in target info and objective, then after it starts we mainly see main agent trace, maybe anomalies, and basic status indicators, like run #'s, what subagents are running, on what chips. Simple tui tho.

For SGLang, the default benchmark command should probably use:

--output-details --cache-report

**agent maintined below this line**

LLM Access setup

- `langchain-openai` ChatGPT OAuth for auth.
- LangGraph should handle the agent loop/state/trace side as much as possible. The TUI can render graph updates and tool events, but we should not build a whole fake trace system at first.
- A few host-owned tools for running benchmarks, logging results, logging anomalies, and enforcing caps.

For auth, we should use `login_chatgpt` from `langchain-openai`.

It runs the ChatGPT OAuth flow with PKCE, starts a local loopback callback server, can open the browser, and persists the token to a file provider. The simple version is: on startup, if the token is missing, run login once, then reuse the stored token. For SSH/headless boxes, use the device login fallback instead of relying on a browser.

The agent/tool setup should be pretty small:

- main agent gets the basic LangChain tools: read files, search the full local codebase, run normal commands, web search, github, coding related stuff, delegate subagents, query explore state, and query SGLang docs
- the github tool should feel similar to the Codex github plugin/skill, not just random web search over github. Read-only issue/PR/code search is enough at first
- subagents do not need github
- raw command running is fine for setup and inspection, but real benchmark experiments should go through the benchmark tool
- main agent gets a LangGraph path for delegating benchmark subagents
- benchmark subagents get runpodctl, remote session tools, `sglang_bench_serve`, `log_experiment`, `log_anomaly`, and query explore state
- anomaly agent should not be a tool right now. The anomaly thing is a separate host job that runs every 30 mins over saved data

When main agent delegates, it should basically just give the subagent a goal. The host knows the agent #, run #, and experiment cap. The prompt does not need that bookkeeping unless we are debugging.

Subagents should not write SQLite directly. They call the narrow logging tools. This keeps the DB boring.

## Main Agent

This is the main agent / coordinator stuff.

Main agent owns the outer loop:

- read the user objective from the TUI
- search/read the full target codebase when it needs implementation context
- query ExploreState before deciding what to try
- query the SGLang docs vector DB when it needs benchmark flags, setup details, or architecture context
- decide what areas are worth poking at
- delegate benchmark subagents with detailed goals
- keep max active subagents at 3
- track which subagent owns which pod
- stop after 3 days max
- shut everything down cleanly when the user hits End or the deadline fires

The main graph can stay small:

`START -> load_state -> think -> maybe_delegate -> maybe_summarize -> think`

Then stop through:

`think -> shutdown_all_pods -> END`

Main does not run SGLang benchmarks directly unless we add that later. Its job is choosing directions, avoiding repeats, reading traces, and deciding when to send more workers.

`delegate_benchmark_subagent` should take one rich `goal` string. No separate `context` field. If the main agent has context, it should write it into the goal.

Main should be allowed to be loose with subagents. The goal can say "try roughly N useful runs", but the host cap is the real hard limit. Subagents can end early if the work is exhausted or setup is sketchy.

Main needs a read-only `query_sglang_docs` tool.

That tool should be backed by a local vector DB of SGLang docs/chunks:

- docs source: SGLang docs and other pinned SGLang reference material we ingest
- embeddings: LangChain embeddings
- storage: local SQLite, preferably separate tables from experiment logging
- query input: natural language question
- output: short relevant chunks with source/path/url and maybe a score

Keep this tool boring. It answers "what does SGLang say about this?" It should not run shell commands, hit github, or mutate state.

ExploreState is separate from the SGLang docs DB:

- SGLang docs DB is static-ish reference material
- ExploreState is what Perferox has tried, seen, logged, and summarized
- full codebase search is for live implementation context, not memory
- both can use embeddings
- both should be queryable by main
- subagents get ExploreState query tools, but not necessarily the whole SGLang docs tool at first

When main stops, it must run a shutdown hook even if the model does not want to:

- set stop requested
- prompt active subagents to wrap up
- stop creating new pods
- stop starting new benchmarks
- tear down every pod started by this main run
- mark leftover runs/subagents as stopped or failed

## Subagents

Subagents are benchmark workers, not anomaly agents.

Each subagent gets a different pod. The shape should be:

1. open a new pod through `runpodctl`
2. wait until SSH works
3. run the default basic setup on the remote machine
4. install SGLang and whatever deps are needed
5. try benchmarks based on the detailed goal from main
6. call `log_experiment` and `log_anomaly` as needed
7. shut down the pod

This should be modeled as a fixed LangGraph, not as one graph node per benchmark attempt.

Rough graph:

`START -> create_pod -> wait_for_ssh -> basic_setup -> benchmark_loop -> wrap_up -> teardown_pod -> END`

If setup fails weirdly, route through:

`basic_setup -> setup_intervention -> basic_setup`

`benchmark_loop` can cycle through tool calls until the subagent ends, hits the cap, hits a deadline, or gets stopped.

Tool access should change by step, using the tool set / `tool_choice` for that node. The setup step gets the setup tool. The benchmark loop gets the benchmark/logging/explore tools. After the successful benchmark count hits the cap, inject a message telling the subagent to wrap up, and do not let it start another benchmark.

The setup step is not fully deterministic, because pods can start from different images / drivers / CUDA state / package state. The default path should still be boring: run the known install/bootstrap commands on the pod and return verbose output. But if setup does not go according to plan, we should interject with a narrow setup intervention step.

`setup_intervention` can inspect the remote machine with limited tools, patch the setup commands, or ask the user what to do. It should not be a general benchmark loop. Once the machine is usable, route back into `basic_setup` or forward into `benchmark_loop`.

Remote execution should use Paramiko. A `RemoteSession` class should own the SSH connection and remote command execution. The LangGraph state should store a session id / pod id, not the live Paramiko object. Remote tools can look up the live session from a host registry.

The SGLang benchmark tool should be one tool with a rich args schema. It should run the serving benchmark command, capture stdout/stderr, parse the useful metrics, and let the agent run `--help` for more context when needed.

The subagent system prompt should say:

- use `runpodctl --help` if it needs more context on pod commands
- use the remote session tools for anything on the pod
- run benchmarks through the SGLang benchmark tool, not random shell once we are in the benchmark loop
- always log useful runs with `log_experiment`
- log weird or human-interesting behavior with `log_anomaly`
- it can deviate a little from the requested permutations if it sees a likely vuln / weak spot
- it can end early if the goal looks exhausted or the pod/setup is not worth pushing

Main should delegate with a single detailed goal. There should not be a separate context field on `delegate_benchmark_subagent`. The goal can include target, suspected weak spots, allowed permutations, rough count, and anything the main agent has already learned.

Main should be lenient about exact counts. The cap is a hard safety limit, not a promise that every subagent must grind all the way to it.

Caps should still be deterministic and host-enforced:

- max active subagents is 3
- each subagent has its own pod
- each subagent has a success cap for benchmark runs
- started benchmarks can also have an attempt cap / deadline so they cannot loop forever
- the main agent has a max runtime of 3 days

When the main agent stops, it should run a shutdown hook:

1. set stop requested
2. inject a shutdown/wrap-up message into active subagents
3. refuse new pod creation and new benchmarks
4. let active benchmark commands finish if reasonable
5. tear down every pod started by this main run
6. mark unfinished runs/subagents as stopped or failed

The shutdown hook should not just trust the model. The host should track which pods were created and tear them down even if an agent is confused.

ExploreState should be shared across the main agent and all subagents.

The simple version is an append-only event log in SQLite:

- `explore_events`: raw events from main/subagents/tools
- `explore_summaries`: compact summaries up to an event id
- embeddings for intents / summaries / useful notes

This is basically the non-destructive summarization pattern. We keep raw history in SQLite for audit/debugging. Summaries are extra events, not replacements. Before each model call, reconstruct the effective context from:

- latest relevant summary
- recent unsummarized events
- embedding hits for similar experiments/intents
- current run/subagent state

Main and subagents should both get tools like:

- `query_explore_state`
- `query_similar_experiments`
- `log_explore_event`

This lets every worker see what has already been tried without copying huge histories into every prompt.

Agent IDs and run IDs should be deterministic:

- agent # starts at 0 and increments by 1 forever
- run # starts at 0 for each agent and increments by 1
- when agent # increases, run # resets
- main agent can be agent 0
- first benchmark subagent can be agent 1, next one agent 2, etc

So the shape is:

`agent 0, run 0`
`agent 0, run 1`
`agent 1, run 0`
`agent 1, run 1`
`agent 2, run 0`

SQLite should enforce:

`unique(agent #, run #)`

The host owns these counters. The model never chooses them.

`experiment_cap` means max successful benchmark runs a subagent is allowed to complete.

This should be enforced in the benchmark tool, not trusted to the agent:

- if stop was requested, refuse to start a new benchmark and tell the agent to wrap up
- count how many successful runs this agent already has
- if successes >= experiment_cap, refuse to start a new benchmark and tell the agent to wrap up
- otherwise assign run #, insert a started run row, run the command, then mark the run finished or failed

Failed/crashed benchmarks get a run row and count against the attempt cap / deadline, but not against the success cap. Invalid input that never starts the benchmark should not count.

If we run subagents concurrently, run assignment has to happen inside a SQLite transaction. Otherwise two calls can both think they got the same run #.


The experiment/logging side should store three tables for now:

- `experiments`
- `runs`
- `anomalies`

For `experiments`, the important things are:

- agent #
- run #
- gpu
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
- gpu
- started_at
- finished_at
- status
- trace_path / trace_ref

For `anomalies`, we need:

- agent #
- run #
- gpu
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

It also needs an End button. End should be a soft stop:

- set `stop_requested = true`
- do not start new benchmarks
- let the current benchmark finish if one is already running
- have tools tell agents to wrap up and summarize

The state machine can be:

`Idle -> Running -> Stopping -> Done`

For SGLang, the default benchmark command should probably use:

`--output-details --cache-report`

## Implementation Plan

First code slice is subagent/runtime infrastructure, not the TUI or the 30 min anomaly worker.

The package should stay small:

```text
perferox/
  db.py              # sqlite schema + writes + cap/id transactions
  state.py           # dataclasses / TypedDicts for graph state
  remote.py          # Paramiko RemoteSession + SessionRegistry
  runpod.py          # runpodctl wrapper
  setup.py           # basic setup commands + setup diagnostics
  bench.py           # SGLang command builder + parser
  tools.py           # LangChain tools
  explore.py         # ExploreState event log + embedding queries
  docs.py            # local SGLang docs vector search
  subagent.py        # benchmark subagent LangGraph
  main_agent.py      # main coordinator graph
  prompts.py         # system prompts
```

Build order:

1. Add `pyproject.toml` with the real deps: `langchain`, `langgraph`, `langchain-openai`, `textual`, `paramiko`, `pydantic`, and probably `numpy` for simple embedding cosine search.
2. Build `db.py`: `runs`, `experiments`, `anomalies`, `explore_events`, `explore_summaries`, `doc_chunks`, and transaction helpers for agent ids / run ids / caps / status updates.
3. Build `remote.py`: `RemoteSession.connect/run/put/close` and `SessionRegistry`. Graph state stores only ids/strings, never Paramiko clients.
4. Build `runpod.py`: `runpodctl_help`, `create_pod`, `wait_for_ssh`, `teardown_pod`, and pod tags for run/agent metadata.
5. Build `setup.py`: `basic_remote_setup(session)` runs the default setup commands and returns logs. Weird setup failures route to `setup_intervention`.
6. Build `bench.py`: one `sglang_bench_serve` args schema, command builder, and parser. Defaults include `--output-details --cache-report`.
7. Build `tools.py`: `sglang_bench_serve`, `log_experiment`, `log_anomaly`, `finish_subagent`, and `query_explore_state`. No github for subagents.
8. Build `subagent.py`: fixed lifecycle graph. Only `benchmark_loop` is agent-flexible. After a successful benchmark, bind only logging/finish tools until `log_experiment`. After success cap, inject wrap-up prompt and remove benchmark tools. Teardown always runs.
9. Build `explore.py`: append-only events, non-destructive summaries, and context reconstruction from latest summary + recent events + embedding hits.
10. Build `docs.py`: local SGLang docs vector DB. Simplest version stores chunks + embedding blobs/json in SQLite and does cosine search in Python. Swap to `sqlite-vec` later only if needed.
11. Build `main_agent.py`: main graph, max 3 active subagents, `delegate_benchmark_subagent(goal, max_successful_benchmarks, deadline)`, `query_sglang_docs`, `query_explore_state`, 3 day deadline, and shutdown hook.

Tests should be few and high signal:

- run id assignment transaction
- success cap vs attempt cap
- SGLang parser on saved sample output
- ExploreState reconstruction
- docs vector query returns nearest chunk

Default tests should not call live RunPod, Paramiko, models, github, ChatGPT OAuth, or cloud GPUs. Use fake sessions and fake `runpodctl` output until we intentionally do an integration run.
