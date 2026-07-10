# Perferox structures

This document describes the structure implemented on `master`. It is an operational map: what runs where, which layer owns each fact, and how work moves through the system.

Perferox is intentionally small. I think of agentic engineering as carefully deciding how much agency to give your agent. The reason this was made, opposed to messing around with codex to get a similar thing done, is to control the agents' agency

## System map

```mermaid
flowchart LR
  user["User"] --> surface["Textual TUI / CLI"]
  surface --> runner["Host runner"]
  runner --> main["Main agent in tmux"]
  main --> source["Persistent ./sglang checkout"]
  main --> workers["Benchmark subagents in tmux"]
  workers --> pod["RunPod / Lambda instance"]
  workers --> ssh["Host-owned SSH session"]
  ssh --> pod

  main <--> db[("SQLite state")]
  workers <--> db
  main --> traces["JSONL traces"]
  workers --> traces
  db --> surface
  traces --> surface

  main -. "native web search" .-> web["Web"]
  workers -. "native web search" .-> web
```

The important split is:

| Model-owned | Host-owned |
| --- | --- |
| hypotheses and exploration direction | agent and run IDs |
| repository setup strategy | attempt caps and stop state |
| benchmark parameters | SQLite writes and transactions |
| whether behavior looks interesting | tmux sessions, trace paths, and SSH objects |
| concise summaries | tool schemas and command execution |

Agents receive goals and immutable constraints, then choose the simplest useful path inside those bounds.

## Entry points and processes

`perferox` exposes four user paths:

- no command opens the Textual dashboard
- `perferox run <objective>` launches the main agent
- `perferox status` prints persisted state
- `perferox end` requests a soft stop

The TUI and CLI both launch `perferox.agent_runner`. The API-key prefix selects RunPod or Lambda, and the runner passes the key to detached tmux processes through a one-use file. Workers expose only the selected provider's environment variable.

The main process uses two roots:

- **runtime root** — the Perferox checkout containing SQLite, traces, `uv`, and worker launch code
- **source root** — a persistent full SGLang clone at `<runtime root>/sglang` used by the main agent's code-reading tools

## Main coordinator

The main agent is responsible for logical reasoning work and finding perf vulnerabilities. This allows us to use cheaper models for subagents, too. 

```mermaid
flowchart TD
  start["Start main process"] --> source["Clone or reuse ./sglang"]
  source --> model["Main model"]
  model -->|"tool call"| tools["Main ToolNode"]
  tools --> model
  model -->|"no local tool call"| wait["Runner waits for SQLite or tmux events"]
  wait -->|"notification or session change"| model
  wait -->|"stop requested and no workers"| done["Exit"]
```

Before each model call, the coordinator receives:

- the user objective
- compact ExplorerState lines
- recent tmux session rows
- accumulated LangGraph messages

Its tools are:

- `bash`, `read_file`, and `search_files` against the SGLang source root
- read-only SQLite queries
- semantic lookup over SGLang `doc_chunks`
- semantic lookup over prior experiment intents
- read/write access to compact ExplorerState
- `delegate_benchmark_subagent`
- native server-side web search

Delegation takes exactly four model-supplied values: `repository`, `commit`, `goal`, and `attempt_cap`. The host validates them, assigns the next `agent_id`, creates trace/goal files, and starts `perferox-agent-<id>` in tmux. At most three subagents may be active.

## Benchmark worker

Each subagent receives one exact repository, commit, goal, and hard attempt cap. The graph changes its local tools by phase while web search remains available during active model phases.

```mermaid
flowchart TD
  start["START"] --> create["Create cloud instance"]
  create -->|"provider CLI / connect SSH"| create_tools["Create-instance tools"]
  create_tools --> create
  create -->|"SSH connected"| setup["Basic setup"]

  setup -->|"remote commands"| setup_tools["Setup tools"]
  setup_tools --> setup
  setup -->|"setup failed"| intervention["Setup intervention"]
  intervention -->|"remote commands"| intervention_tools["Intervention tools"]
  intervention_tools --> intervention
  intervention -->|"retry"| setup

  setup -->|"target ready"| bench["Benchmark loop"]
  bench -->|"run / log tools"| bench_tools["Benchmark tools"]
  bench_tools --> bench
  bench -->|"cap, stop, or finished"| wrap["Wrap up"]
  intervention -->|"unrecoverable"| wrap
  wrap --> finish["END"]
```

The normal setup path is:

1. choose a RunPod or Lambda environment
2. optionally use a container when it clearly reduces setup work
3. clone the delegated repository into `/workspace/target`
4. check out the exact commit in detached HEAD state
5. verify it with `git rev-parse HEAD`
6. follow the repository's own build instructions

The container is a suggestion, not a requirement. For SGLang, the prompt points workers to `lmsysorg/sglang` image tags as a useful starting point.

Worker tools are deliberately phase-scoped:

| Phase | Mutating capabilities |
| --- | --- |
| create instance | local `runpodctl` or `lambda-labs`, connect host SSH session |
| setup / intervention | remote shell over the registered SSH session |
| benchmark | remote shell, structured SGLang benchmark, log experiment, log anomaly |
| wrap-up | write one summary notification to SQLite |

The worker stores only messages, `agent_id`, and its final summary in LangGraph state. Live SSH clients stay in a host `SessionRegistry`, never in graph state or traces.

## One benchmark attempt

Real experiments go through `sglang_bench_serving`; raw remote commands are for setup and inspection.

```mermaid
flowchart LR
  args["Typed BenchServingArgs"] --> command["Normalize SGLang command"]
  command --> tx["BEGIN IMMEDIATE"]
  tx --> guard{"stop or cap reached?"}
  guard -->|"yes"| refuse["Refuse new run"]
  guard -->|"no"| row["Insert started run + assign run_id"]
  row --> remote["Execute over SSH"]
  remote -->|"failed"| failed["Mark run failed"]
  remote -->|"succeeded"| parse["Parse benchmark metrics"]
  parse --> experiment["Log experiment + intent embedding"]
  experiment --> anomaly["Optionally log anomaly"]
  failed --> notify["Notify main"]
  experiment --> notify
  anomaly --> notify
```

Started failures count against the cap. Invalid arguments do not, because no run row is created. SQLite serializes run-number assignment and enforces the cap in the same transaction.

## Durable state

SQLite is the source of truth; prompts and message history are not bookkeeping systems.

| Table | Purpose |
| --- | --- |
| `runs` | every started benchmark, command hash, timing, trace, and failure state |
| `experiments` | successful normalized metrics plus human-readable intent and embedding |
| `anomalies` | human-readable surprising behavior tied to a run |
| `agent_sessions` | main/subagent tmux identity and lifecycle status |
| `main_notifications` | durable wakeups for run, experiment, anomaly, and summary events |
| `explorer_state_lines` | compact append-only exploration memory |
| `doc_chunks` | locally ingested SGLang reference text and embeddings |

`(agent_id, run_id)` is the run identity. `run_id` starts at zero for each agent. The command `exact_hash` is unique, preventing the same normalized benchmark command from being started twice.

## Soft stop

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Running: START
  Running --> Stopping: END / perferox end
  Stopping --> Done: active workers finish
  Done --> Running: new objective
```

The End action changes running `agent_sessions` rows to `ending`. After that:

- the main agent refuses new delegation
- `start_benchmark_run` refuses new attempts
- workers observe the stop flag and route to wrap-up
- an already-running remote benchmark is allowed to finish
- the main process exits after no worker sessions remain

The host does not rely on a model voluntarily honoring the stop request, but we do encourage the agent to stop after we hit the cap

## Modules

| Module | Responsibility |
| --- | --- |
| `cli.py` | CLI routing for TUI, run, status, and end |
| `tui.py` | OAuth gate, live dashboard, launch, and soft-stop controls |
| `agent_runner.py` | tmux process entry points, persistent SGLang workspace, traces, and wakeups |
| `main_agent.py` | coordinator graph, research tools, ExplorerState, and delegation |
| `subagent.py` | fixed worker lifecycle graph and final summary notification |
| `tools.py` | local/remote execution and narrow host-owned LangChain tools |
| `bench.py` | typed SGLang serving arguments, command generation, and metric parsing |
| `db.py` / `init-db.sql` | transactions, IDs, caps, persistence, embeddings, and notifications |
| `remote.py` | Paramiko SSH session and in-process session registry |
| `auth.py` | persisted ChatGPT OAuth, cloud-key validation and one-use handoff |
| `prompts.py` | provider-specific instance creation and worker constraints |
| `packages/lambda-labs/lambda_labs.py` | small Lambda Cloud CLI used by workers |
