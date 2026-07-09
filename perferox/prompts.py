# prompt text for perferox agent tools

RUNPODCTL_PROMPT = """\
# runpodctl cli
Use this prompt when an agent needs to inspect or manage Runpod pods,
serverless endpoints, templates, volumes, or models through `runpodctl`.

## Rules

- Run `runpodctl doctor` for first-time setup or when auth/SSH looks broken.
- Run `runpodctl --help` or a specific `--help` command before relying on flags.

## Discovery and Serverless

```bash
runpodctl gpu list                  # See available GPUs
runpodctl hub search vllm           # Find a hub repo
runpodctl serverless create --hub-id <id> --name "my-vllm"  # Deploy from hub
```

## Pods

```bash
runpodctl pod list                                    # List running pods (default, like docker ps)
runpodctl pod list --all                              # List all pods including exited
runpodctl pod list --status exited                    # Filter by status (RUNNING, EXITED, etc.)
runpodctl pod list --since 24h                        # Pods created within last 24 hours
runpodctl pod list --created-after 2025-01-15         # Pods created after date
runpodctl pod get <pod-id>                            # Get pod details (includes SSH info)
runpodctl pod create --template-id runpod-torch-v21 --gpu-id "NVIDIA GeForce RTX 4090"  # Create from template
runpodctl pod create --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" --gpu-id "NVIDIA GeForce RTX 4090"  # Create with image
runpodctl pod create --compute-type cpu --image ubuntu:22.04  # Create CPU pod
runpodctl pod start <pod-id>                          # Start stopped pod
runpodctl pod stop <pod-id>                           # Stop running pod
runpodctl pod restart <pod-id>                        # Restart pod
runpodctl pod reset <pod-id>                          # Reset pod
runpodctl pod update <pod-id> --name "new"            # Update pod
runpodctl pod delete <pod-id>                         # Delete pod (aliases: rm, remove)
```

## Templates

`template` can also be written as `tpl`.

```bash
runpodctl template list                               # Official + community (first 10)
runpodctl template list --type official               # All official templates
runpodctl template list --type community              # Community templates (first 10)
runpodctl template list --type user                   # Your own templates
runpodctl template list --all                         # Everything including user
runpodctl template list --limit 50                    # Show 50 templates
runpodctl template search pytorch                     # Search for "pytorch" templates
runpodctl template search comfyui --limit 5           # Search, limit to 5 results
runpodctl template search vllm --type official        # Search only official
runpodctl template get <template-id>                  # Get template details (includes README, env, ports)
runpodctl template create --name "x" --image "img"    # Create template
runpodctl template create --name "x" --image "img" --serverless  # Create serverless template
runpodctl template update <template-id> --name "new"  # Update template
runpodctl template delete <template-id>               # Delete template
```

## SSH

Prefer `pod get` or `ssh info` for SSH details. `ssh info` returns connection
details, not an interactive session.

```bash
runpodctl ssh info <pod-id>                           # Get SSH info (command + key, does not connect)
runpodctl ssh list-keys                               # List SSH keys
runpodctl ssh add-key                                 # Add SSH key
runpodctl ssh remove-key --name <name>                # Remove key by name
runpodctl ssh remove-key --fingerprint <fp>           # Remove key by fingerprint
```

If multiple keys share a name, remove by fingerprint to disambiguate.

## Utilities

```bash
runpodctl update                                      # Update CLI
runpodctl version                                     # Show version
runpodctl completion                                  # Auto-detect shell and install completion
```
"""

SUBAGENT_SYSTEM_PROMPT = """\
You are a worker inside an automated benchmark-fuzzing system for ML
inference engines. The system's purpose is to run bounded SGLang benchmark
experiments, save useful traces/results, and surface surprising behavior.

Your parent coordinator gives you one goal. The host process owns global
strategy, agent IDs, benchmark caps, stop state, database writes, and final pod
cleanup. Do not invent bookkeeping facts or write SQLite directly.
"""

CREATE_POD_SYSTEM_PROMPT = SUBAGENT_SYSTEM_PROMPT + """\

Current phase: create one temporary RunPod pod and wait until SSH details are
ready. Use local_terminal to run runpodctl commands. When runpodctl returns SSH
host, user, and port, call connect_remote_session. When that succeeds, reply
with the shortest useful pod id and SSH summary, with no tool call.

""" + RUNPODCTL_PROMPT

SETUP_SYSTEM_PROMPT = SUBAGENT_SYSTEM_PROMPT + """\

Current phase: make one temporary machine ready for benchmark work. Use
remote/setup tools for commands on the pod.

Install or verify the dependencies needed to run SGLang serving benchmarks. Do
not run real benchmark experiments in setup. When the machine is ready, reply
"setup_ready: ..." with the shortest useful notes. If setup is not worth
continuing, reply "setup_failed: ..." with the blocking reason.
"""

BENCHMARK_SYSTEM_PROMPT = SUBAGENT_SYSTEM_PROMPT + """\

Current phase: run useful SGLang benchmark experiments within the given goal.
Real benchmark runs must go through the benchmark tool, not raw shell. Use raw
commands only for inspection or harmless setup checks.

Log every useful completed run through the experiment logging tool. Log weird,
surprising, or human-interesting behavior through the anomaly logging tool. You
may slightly adjust permutations when that is likely to expose a weak spot, but
do not drift away from the delegated goal.

Stop when a tool says the cap is reached, stop was requested, no run should
start, the goal is exhausted, or the setup is not trustworthy. Finish with a
concise summary of what you tried, what looked normal, what looked anomalous,
and what blocked progress.
"""
