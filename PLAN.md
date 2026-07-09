This project is meant to be something on the order of a fuzzer, except for inference engines and ML sys perf stuff. The initial target is SGLang.

The goal is to run a custom, minimal agent in a loop, to make it stress test the extents of a program. This will be done by making the agent spin up different (w/ often odd setups) cloud GPUs and clusters, and trying to see how perf and parity regressed across git commits, how it performs on odd or ill-maintained settings, how it performs on AMD chips, the difference between backends and so forth. Similar to how engineers woud write programs of all sorts, or run vLLM/SGlang configs of all different types, we emulate that.

The reason this this is so doable is because we only have to track a few metrics to measure for anomalies (cosine sim, TPS, TTFT, cache hit rate, startup s, etc.)

The agent will be given minimal tools, and will definitely have to compact information in a unique manner, such that we don't retry things. Maybe we can have a deterministic write and check tool. The agent will also be given access to runpod/lambda to spin up stuff to run code on.

U might assume we can do something like this through a general coding agent. The main reason this isn't the case is because of compaction. We do not need to store most context throughout, only certain things, in certain ways, and another is creating dedicated subagents. We get to craft subagents for a specific use case. It also helps that we can provide heavy guidance on how to set up pods to each subagent, and because we use a custom harness, we can track experiments in a much more organized manner.

Goal is < 2 kLOC. edit: we got to ~1.8 kLOC!
