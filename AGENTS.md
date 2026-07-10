# Repository Guidelines

Naman owns this repository.

## Project Objective

Perferox is a small agentic fuzzer/stress tester for inference engines and ML systems performance. The first target is SGLang.

The system should run bounded benchmark experiments, preserve useful traces, detect anomalies, and avoid repeating the same intent. Keep the project small. The target is still roughly `< 5 kLOC`.

## Source of Truth

- `structures.md` is the current architecture source of truth.
- `README.md` is project context, but it may lag behind recent stack decisions.
- If `README.md` and `structures.md` conflict, follow `structures.md` and surface the conflict before broad changes.
- Do not create new docs unless they remove real ambiguity.

## Current Stack

- Language: Python.
- TUI: Textual.
- Agent/tool wrappers: LangChain.
- Agent state and routing: LangGraph.
- Local storage: SQLite.
- OpenAI auth: `langchain-openai` ChatGPT OAuth through `login_chatgpt`.
- Initial target: SGLang benchmarks using `--output-details --cache-report`.

If Python packaging is needed, prefer `uv` unless the repo later chooses something else.

## Coding style

- Prefer plain data plus small functions over class hierarchies.
- Prefer one sharp abstraction over many soft abstractions.
- Keep state explicit and inspectable.
- Use table-driven rules where possible: lists, dicts, small enums, dataclasses, and simple dispatch tables.
- Do not add a framework around a framework. LangGraph is the workflow layer.
- Do not add an abstraction for one caller.
- Keep hot-path records small. Use `dataclass(slots=True)` for internal records when it is natural.
- Prefer deterministic transforms over model judgment for bookkeeping.
- Let SQLite constraints and transactions enforce facts. Do not trust prompts to count correctly.
- Use stdlib tools before dependencies when the stdlib version is clear.
- Clever code is allowed only at pressure points. Most code should be boring.
- Give agents the goal and immutable constraints, then let them choose the basic execution path.

The main pattern should be:

```text
normalize intent -> check repeat -> run benchmark -> parse output -> log experiment/anomaly -> update graph state
```

## Agent Architecture

- Main agent plans and delegates benchmark work.
- Benchmark subagents run experiments and log results.
- Do not expose an anomaly agent as a tool right now.
- If anomaly analysis becomes separate later, run it as a host-driven pass over saved results.
- Main agent should delegate mostly with a goal. The host owns agent IDs, run IDs, caps, and stop state.
- Use LangGraph state, nodes, conditional edges, and streamed updates for inspectable workflows.
- Let LangGraph handle loop/state/trace as much as possible. Do not build a fake trace system first.

## Tools

All agents may have basic read-only or low-risk tools:

- `read_file`
- `search_files`
- `run_command`
- `web_search`

The GitHub tool should feel like the Codex GitHub connector: repo-aware, read-only issue/PR/code search at first. It should not be random web search over GitHub.

Benchmark subagents get the mutating tools:

- `run_benchmark`
- `log_experiment`
- `log_anomaly`

Rules:

- Real experiments must go through `run_benchmark`, not raw `run_command`.
- Agents must not write SQLite directly.
- All SQLite writes go through narrow host-owned tools.
- Tool outputs should tell the agent what changed and what remains.

## Deterministic IDs and Caps

- `agent #` starts at 0 and increments by 1 forever.
- `run #` starts at 0 for each agent and increments by 1.
- When `agent #` increases, `run #` resets.
- The model never chooses either ID.
- SQLite should enforce `unique(agent #, run #)`.

`experiment_cap` is the maximum number of benchmark commands a subagent may run.

`run_benchmark` enforces the cap:

- If `stop_requested` is set, refuse to start a new benchmark and tell the agent to wrap up.
- Count existing runs for the agent inside a SQLite transaction.
- If the cap is reached, refuse to start a new benchmark and tell the agent to wrap up.
- Otherwise assign the next run number, insert a started run row, run the command, then mark it finished or failed.

Failed or crashed benchmarks count once they started. Invalid input that never starts a benchmark does not count.

## Database Rules

Use SQLite as the local source of truth.

Initial tables:

- `experiments`
- `runs`
- `anomalies`

Keep schemas minimal. Add fields only when something reads them, displays them, validates them, or uses them for repeat detection.

Do not store duplicate semantic fields. If two fields mean the same thing, keep one.

Repeat checks should use:

- `exact_hash` for exact experiment specs.
- `intent_key` plus embedding search for semantically similar experiments.

`intent_key` should stay human readable and searchable.

## TUI Rules

Use Textual to build the actual working interface, not a marketing page.

First screen should be the tool:

- target info
- objective/goal
- run controls
- main trace/status
- anomalies
- subagent/run counters

The TUI needs an End button.

End is a soft stop:

- set `stop_requested = true`
- do not start new benchmarks
- let the current benchmark finish
- have tools tell agents to wrap up and summarize

Basic state machine:

```text
Idle -> Running -> Stopping -> Done
```

## Auth and Secrets

- Use `login_chatgpt` for local ChatGPT OAuth when a token is missing.
- Reuse the persisted token after login.
- Use the device-flow fallback on SSH/headless machines.
- Never commit tokens, API keys, OAuth stores, `.env` files, benchmark credentials, or cloud credentials.

## Coding Rules

- Every function should have a short purpose docstring or comment. Keep comments useful.
- Add comments for non-trivial logic, especially cap enforcement, transaction boundaries, command normalization, and output parsing.
- Prefer clear intermediate variables over dense call chains.
- Avoid broad refactors.
- Do not edit adjacent code just to make it prettier.
- Delete code only when the deletion directly follows from the requested change.
- If a simpler one-line solution works, use it.
- Avoid defensive layers that do not have a concrete failure mode.
- Keep modules focused:
  - graph/state orchestration
  - tools
  - db
  - benchmark parsing
  - TUI
  - auth

## Testing and Validation

Write few tests, but make them high signal.

Prioritize tests for:

- command normalization
- exact hash stability
- cap enforcement
- concurrent run ID assignment
- SGLang output parsing
- anomaly logging

Do not add broad tests for trivial getters or obvious dataclass construction.

Do not make live cloud, GitHub, browser, ChatGPT OAuth, or expensive model calls part of default tests.

## External and Cloud Safety

- Do not provision RunPod, Lambda, cloud GPUs, or clusters unless the user explicitly asks in that turn.
- Do not run destructive benchmark commands unless they are clearly part of the requested experiment.
- Prefer read-only GitHub behavior until write behavior is explicitly requested.
- Treat paid model calls and cloud GPU time as externally visible cost.

## Git Safety

- Do not commit or push unless the user explicitly asks.
- Do not use destructive git commands unless explicitly requested.
- Do not use `git add .`; stage explicit paths.
- Preserve user changes. If a file changed unexpectedly, re-read it before editing.

## Quality Bar

The best Perferox code should feel extremely high quality:

- small files
- sharp data structures
- deterministic host-owned state
- explicit pipelines
- narrow tools
- easy replay
- no fake enterprise architecture

When in doubt, write less code.
