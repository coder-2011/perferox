<h1 align="center">perferox</h1>

<p align="center">
  <a href="perferox/perferox-tui-demo.mp4">
    <img src="perferox/perferox-tui-demo.gif" alt="Perferox TUI demo" width="900">
  </a>
</p>


Perferox is an intelligent performance fuzzer for AI systems software, to try to spot performance bugs in prod systems. It can be used as a CI, always running agent, or extensive performance testing for package releases.

It is geared towards:
- Kernel libraries
- RL infra projects
- Distributed training libraries
- Model serving engines

## What it does
Perferox is an agentic loop that tests the absolute edges of a system, by using it in increasingly niche ways, looking for recent PRs that regressed perf or capabilities, trying different combinations of settings, trying less-used/accessible chipsets, etc. in an effort of finding hidden regressions that many users would not report, or that often go unnoticed. 

Not everyone opens a github issue when something goes wrong...

## How it works

A main agent spawns subagents which each live in persistent tmux sessions, and spin up independent VMs through a neocloud provider

Based off the instructions of the main agent, tries attacking a certain *vulnerability* the main agent thinks is worth while prodding about, and comes back with the results.

Then, it updates a local DB with detailed information on experiments ran, and points out when it notices an anomaly.

All this information is easily visible through the TUI, and can also quickly be seen via CLI

### Extra Info

As of Jul. 9th, this is completely geared towards testing on runpod, and is built for SGLang. In the future, I plan on making it generalized such that it is very easy to point an agent at perferox source code and it can customize it for whatever project you are working on. 

I also plan on adding support for other major projects (vLLM, DeepSpeed, flashinfer, etc.) and other neoclouds(lambda, verda, and most major ones w/ a CLI/MCP)

As of Jul. 9th, it is clearly pre-beta right now
