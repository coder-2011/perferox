This project is meant to be something on the order of a fuzzer, except for inference engines and ML sys perf stuff. The initial target is SGLang.

The goal is to run a custom, minimal agent in a loop, to make it stress test the extents of a program. This will be done by making the agent spin up different (w/ often odd setups) cloud GPUs and clusters, and trying to see how perf and parity regressed across git commits, how it performs on odd or ill-maintained settings, how it performs on AMD chips, the difference between backends and so forth. Similar to how engineers woud write programs of all sorts, or run vLLM/SGlang configs of all different types, we emulate that.

The reason this this is so doable is because we only have to track a few metrics to measure for anomalies (cosine sim, TPS, TTFT, cache hit rate, and a few others I am not thinking of rn)

The agent will be given minimal tools, and will definitely have to compact information in a unique manner, such that we don't retry things. Maybe we can have a deterministic write and check tool. The agent will also be given access to runpod/lambda to spin up stuff to run code on.

Things that need to be built for this:

- A minimal agent (potentially a fork of pi) that has access to deploy subagents on different VMs. This one can decide overall direction, such as whether to check for git history regressions, to try little-maintained backends, and so forth. The highest granularity it will work at is designating what style of new parts of codebase to run on a new pod, and how hard to push that specific sector.
- Skills/MD files for how to use runpod, how to boostrap VMs, diff docker setups accessible to SGLang or wtv we are stress testing, the general outline of wtv codebase is being used, how to use different tools, sysprompt.md, etc.
- A tool to write results and a simple mechanism for checking if we are repeating experiments. This can probably be fully deterministic.
- A tool to call out anomalies and store them in a seperate file, which is user-friendly.
- Potentially a seperate api call to go over all results, and check for anomalies.
- A simple SQLite DB to store basic information, and probably also agent trace. So we store `anomalies`, `experiments`, and `runs`, each as seperate tables.

We will probably write this in either python or rust. I haven't messed around w/ Rust that much, so maybe rust, just for kicks.

Goal is < 5 kLOC
