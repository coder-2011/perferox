# prompt text for perferox agent tools

RUNPODCTL_PROMPT = """\
# runpodctl cli
Runpod CLI to manage GPU workloads.

Use this prompt when an agent needs to inspect or manage Runpod pods,
serverless endpoints, templates, volumes, or models through `runpodctl`.

## Rules

- Run `runpodctl doctor` for first-time setup or when auth/SSH looks broken.
- ** Run `runpodctl --help` or a specific `--help` command before relying on flags. **
- For SSH details, prefer `runpodctl pod get <pod-id>` or
  `runpodctl ssh info <pod-id>`.
- `runpodctl ssh info` returns connection details, not an interactive session.

## Quick Start

```bash
runpodctl doctor                    # First time setup (API key + SSH)
runpodctl --help                    # See current top-level commands
runpodctl pod create --help         # Inspect exact current flags before creating
runpodctl gpu list                  # See available GPUs
runpodctl hub search vllm           # Find a hub repo
runpodctl serverless create --hub-id <id> --name "my-vllm"  # Deploy from hub
runpodctl template search pytorch   # Find a template
runpodctl pod create --template-id runpod-torch-v21 --gpu-id "NVIDIA GeForce RTX 4090"  # Create from template
runpodctl pod list                  # List your pods```

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
runpodctl pod delete <pod-id>                         # Delete pod (aliases: rm, remove)```

For exact pod flags, run:

```bash
runpodctl pod <action> --help
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
runpodctl doctor                                      # Diagnose and fix CLI issues
runpodctl update                                      # Update CLI
runpodctl version                                     # Show version
runpodctl completion                                  # Auto-detect shell and install completion
```
"""
