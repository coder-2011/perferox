"""First-run setup: pick a model provider, authenticate it, pick a GPU cloud.

Onboarding writes one profile to `~/.perferox/config.json` and, when a key is
pasted rather than exported, one credential file next to it. Everything it asks
is answerable from the provider table, so a new backend shows up in this flow
without touching it.
"""

from __future__ import annotations

import os

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from perferox import auth, providers
from perferox.providers import KEY, LOCAL, OAUTH, Provider, Settings

BRAND = "#fabd2f"
DIM = "#928374"
CLOUD_LABELS = {"runpod": "RunPod", "lambda": "Lambda", "modal": "Modal"}
GROUPS = ((OAUTH, "Sign in with an account"), (KEY, "Bring an API key"), (LOCAL, "Run it on this machine"))


def needs_onboarding() -> bool:
  """Report whether the user has never saved a Perferox profile."""
  return providers.load_settings() is None


def _provider_menu(console: Console) -> list[Provider]:
  """Print every provider as a numbered row and return them in the printed order."""
  ordered: list[Provider] = []
  table = Table("#", "Provider", "Detail", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for auth_kind, heading in GROUPS:
    group = providers.by_auth(auth_kind)
    if not group:
      continue
    table.add_section()
    table.add_row("", Text(heading, style=f"bold {DIM}"), "")
    for provider in group:
      ordered.append(provider)
      ready = " [green]ready[/]" if auth.auth_ready(provider) else ""
      table.add_row(str(len(ordered)), f"{provider.label}{ready}", Text(provider.detail, style=DIM))
  console.print(table)
  return ordered


def _pick_provider(console: Console) -> Provider:
  """Ask which backend Perferox should send its reasoning to."""
  ordered = _provider_menu(console)
  default = next((index for index, provider in enumerate(ordered, 1) if auth.auth_ready(provider)), 1)
  while True:
    choice = IntPrompt.ask("Provider", default=default, console=console)
    if 1 <= choice <= len(ordered):
      return ordered[choice - 1]
    console.print(f"[red]pick a number between 1 and {len(ordered)}[/]")


def _pick_model(console: Console, provider: Provider) -> str:
  """Offer the provider's known model tags and accept a custom one."""
  table = Table("#", "Model", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for index, model in enumerate(provider.models, 1):
    table.add_row(str(index), model)
  table.add_row(str(len(provider.models) + 1), Text("something else", style=DIM))
  console.print(table)
  while True:
    choice = IntPrompt.ask("Model", default=1, console=console)
    if choice == len(provider.models) + 1:
      typed = Prompt.ask("Model tag", default=provider.default_model, console=console).strip()
      if typed:
        return typed
      continue
    if 1 <= choice <= len(provider.models):
      return provider.models[choice - 1]
    console.print(f"[red]pick a number between 1 and {len(provider.models) + 1}[/]")


def _authenticate(console: Console, provider: Provider) -> bool:
  """Sign the provider in or collect its key; return whether it ended up usable."""
  if auth.auth_ready(provider):
    console.print(f"[green]{provider.label} is already authenticated.[/]")
    return True
  if provider.auth == LOCAL:
    return True
  if provider.auth == OAUTH:
    console.print(f"Signing in to {provider.label}.")
    try:
      auth.ensure_auth(provider)
    except Exception as exc:  # noqa: BLE001 - any sign-in failure is reported, not raised
      console.print(f"[red]sign-in failed:[/] {type(exc).__name__}: {exc}")
      return False
    console.print(f"[green]signed in to {provider.label}.[/]")
    return True

  if provider.account_env and not providers.account_id(provider):
    account = Prompt.ask(f"{provider.label} account id", console=console).strip()
    if not account:
      console.print(f"[yellow]skipped; set {provider.account_env} before running.[/]")
      return False
    providers.save_credential(f"{provider.name}_account", account)
  console.print(Text(f"Paste a key to store it in {providers.CREDENTIALS_PATH}, or leave it blank to read {provider.key_env} at run time.", style=DIM))
  key = Prompt.ask(f"{provider.label} API key", password=True, default="", show_default=False, console=console).strip()
  if key:
    providers.save_credential(provider.name, key)
  return auth.auth_ready(provider)


def _cloud_status(name: str) -> tuple[bool, str]:
  """Report whether one GPU cloud is already usable and why."""
  if name == "modal":
    try:
      auth.modal_cloud_key()
    except ValueError as exc:
      return False, str(exc)
    return True, "profile or token variables found"
  env_name = "LAMBDA_API_KEY" if name == "lambda" else "RUNPOD_API_KEY"
  key = os.environ.get(env_name, "")
  if not key:
    return False, f"{env_name} not set; Perferox will ask when a run starts"
  try:
    auth.cloud_provider(key)
  except ValueError as exc:
    return False, str(exc)
  return True, f"{env_name} looks valid"


def _pick_cloud(console: Console) -> str:
  """Ask which GPU cloud benchmark workers should rent instances from."""
  table = Table("#", "Cloud", "State", box=box.SIMPLE_HEAVY, header_style=f"bold {BRAND}")
  for index, name in enumerate(auth.CLOUD_PROVIDERS, 1):
    ready, detail = _cloud_status(name)
    table.add_row(str(index), CLOUD_LABELS[name], Text(detail, style="green" if ready else DIM))
  console.print(table)
  while True:
    choice = IntPrompt.ask("Cloud", default=1, console=console)
    if 1 <= choice <= len(auth.CLOUD_PROVIDERS):
      return auth.CLOUD_PROVIDERS[choice - 1]
    console.print(f"[red]pick a number between 1 and {len(auth.CLOUD_PROVIDERS)}[/]")


def run(console: Console) -> Settings | None:
  """Walk the whole setup and save the profile; return None when the user backs out."""
  console.print(Panel.fit(Text("Perferox setup", style=f"bold {BRAND}"), border_style=BRAND))
  console.print(Text("Pick the model that plans and reads code, then the cloud that runs the benchmarks.\n", style=DIM))

  provider = _pick_provider(console)
  if not _authenticate(console, provider) and not Confirm.ask("Save this provider anyway?", default=False, console=console):
    console.print(Text("nothing saved", style=DIM))
    return None
  model = _pick_model(console, provider)
  cloud = _pick_cloud(console)

  settings = Settings(provider=provider.name, model=model, cloud=cloud)
  providers.save_settings(settings)
  summary = Table.grid(padding=(0, 2))
  summary.add_row(Text("model", style=DIM), f"{settings.provider}:{settings.model}")
  summary.add_row(Text("cloud", style=DIM), CLOUD_LABELS[settings.cloud])
  summary.add_row(Text("profile", style=DIM), str(providers.CONFIG_PATH))
  console.print(Panel(summary, title="[bold]Ready[/]", border_style="green"))
  console.print(Text("start a run with:  perferox run \"find a regression in the radix cache\"\n", style=DIM))
  return settings
