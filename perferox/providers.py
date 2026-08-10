"""Table-driven model provider registry plus the on-disk Perferox profile.

Every backend Perferox can talk to is one row in `PROVIDERS`. Adding a provider
should mean adding a row, not editing the CLI, the TUI, and the agent hosts.
The selected provider, model, and any pasted API keys live under `~/.perferox`
so detached tmux workers read the same choice the user made in the CLI or TUI.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path("~/.perferox").expanduser()
CONFIG_PATH = CONFIG_DIR / "config.json"
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"

# How a provider proves who you are: an account sign-in, a pasted/exported API
# key, or nothing at all because the server runs on this machine.
OAUTH = "oauth"
KEY = "key"
LOCAL = "local"

# Substituted into a base URL that is scoped to an account (Cloudflare).
ACCOUNT_PLACEHOLDER = "{account_id}"


@dataclass(frozen=True, slots=True)
class Provider:
  """One model backend: how it authenticates, where it lives, what it serves."""

  name: str
  label: str
  auth: str
  detail: str
  models: tuple[str, ...]
  base_url: str = ""
  key_env: str = ""
  account_env: str = ""

  @property
  def default_model(self) -> str:
    """Return the first model tag, which every picker offers as the default."""
    return self.models[0]


# Ordered for the onboarding menu: sign-ins first because they need no key,
# then the direct API-key vendors, then OpenAI-compatible clouds, then local.
PROVIDERS: tuple[Provider, ...] = (
  Provider(
    name="chatgpt",
    label="ChatGPT account sign-in",
    auth=OAUTH,
    detail="use a ChatGPT subscription, no API key",
    models=("gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
  ),
  Provider(
    name="grok",
    label="xAI account sign-in",
    auth=OAUTH,
    detail="use an xAI account, no API key",
    models=("grok-4.5", "grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1"),
    base_url="https://api.x.ai/v1",
  ),
  Provider(
    name="openai",
    label="OpenAI",
    auth=KEY,
    detail="gpt-5.6 family via OPENAI_API_KEY",
    models=("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.3-codex"),
    base_url="https://api.openai.com/v1",
    key_env="OPENAI_API_KEY",
  ),
  Provider(
    name="anthropic",
    label="Anthropic (Claude)",
    auth=KEY,
    detail="Claude via ANTHROPIC_API_KEY",
    models=("claude-fable-5", "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"),
    base_url="https://api.anthropic.com/v1",
    key_env="ANTHROPIC_API_KEY",
  ),
  Provider(
    name="xai",
    label="xAI (Grok) API key",
    auth=KEY,
    detail="grok-4.5 via XAI_API_KEY",
    models=("grok-4.5", "grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1"),
    base_url="https://api.x.ai/v1",
    key_env="XAI_API_KEY",
  ),
  Provider(
    name="openrouter",
    label="OpenRouter",
    auth=KEY,
    detail="hundreds of models via OPENROUTER_API_KEY",
    models=("openrouter/auto",),
    base_url="https://openrouter.ai/api/v1",
    key_env="OPENROUTER_API_KEY",
  ),
  Provider(
    name="cloudflare",
    label="Cloudflare Workers AI",
    auth=KEY,
    detail="GLM 5.2 via CLOUDFLARE_API_TOKEN and an account id",
    models=("@cf/zai-org/glm-5.2", "@cf/zai-org/glm-4.7-flash", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/qwen/qwen2.5-coder-32b-instruct"),
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_PLACEHOLDER}/ai/v1",
    key_env="CLOUDFLARE_API_TOKEN",
    account_env="CLOUDFLARE_ACCOUNT_ID",
  ),
  Provider(
    name="gemini",
    label="Google Gemini",
    auth=KEY,
    detail="gemini-3.5-flash via GEMINI_API_KEY",
    models=("gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    key_env="GEMINI_API_KEY",
  ),
  Provider(
    name="deepseek",
    label="DeepSeek",
    auth=KEY,
    detail="deepseek-v4 via DEEPSEEK_API_KEY",
    models=("deepseek-v4-pro", "deepseek-v4-flash"),
    base_url="https://api.deepseek.com/v1",
    key_env="DEEPSEEK_API_KEY",
  ),
  Provider(
    name="groq",
    label="Groq",
    auth=KEY,
    detail="fast open-model inference via GROQ_API_KEY",
    models=("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3.6-27b"),
    base_url="https://api.groq.com/openai/v1",
    key_env="GROQ_API_KEY",
  ),
  Provider(
    name="mistral",
    label="Mistral",
    auth=KEY,
    detail="mistral-medium and devstral via MISTRAL_API_KEY",
    models=("mistral-medium-latest", "mistral-large-latest", "devstral-2512"),
    base_url="https://api.mistral.ai/v1",
    key_env="MISTRAL_API_KEY",
  ),
  Provider(
    name="moonshot",
    label="Moonshot AI",
    auth=KEY,
    detail="Kimi K3 via MOONSHOT_API_KEY",
    models=("kimi-k3",),
    base_url="https://api.moonshot.ai/v1",
    key_env="MOONSHOT_API_KEY",
  ),
  Provider(
    name="zai",
    label="Z.AI",
    auth=KEY,
    detail="GLM 5.2 via ZAI_API_KEY",
    models=("glm-5.2", "glm-5.1", "glm-5"),
    base_url="https://api.z.ai/api/paas/v4",
    key_env="ZAI_API_KEY",
  ),
  Provider(
    name="minimax",
    label="MiniMax",
    auth=KEY,
    detail="MiniMax M2.7 via MINIMAX_API_KEY",
    models=("minimax-m2.7",),
    base_url="https://api.minimax.io/v1",
    key_env="MINIMAX_API_KEY",
  ),
  Provider(
    name="together",
    label="Together AI",
    auth=KEY,
    detail="open models via TOGETHER_API_KEY",
    models=("openai/gpt-oss-120b",),
    base_url="https://api.together.xyz/v1",
    key_env="TOGETHER_API_KEY",
  ),
  Provider(
    name="fireworks",
    label="Fireworks AI",
    auth=KEY,
    detail="open models via FIREWORKS_API_KEY",
    models=("accounts/fireworks/models/gpt-oss-120b",),
    base_url="https://api.fireworks.ai/inference/v1",
    key_env="FIREWORKS_API_KEY",
  ),
  Provider(
    name="cerebras",
    label="Cerebras",
    auth=KEY,
    detail="ultra-fast inference via CEREBRAS_API_KEY",
    models=("gpt-oss-120b",),
    base_url="https://api.cerebras.ai/v1",
    key_env="CEREBRAS_API_KEY",
  ),
  Provider(
    name="ollama",
    label="Ollama (local)",
    auth=LOCAL,
    detail="a model already pulled on this machine",
    models=("qwen3.6:35b", "qwen3.6:27b", "qwen3.5:9b", "qwen3.5:4b"),
    base_url="http://127.0.0.1:11434/v1",
  ),
  Provider(
    name="llamacpp",
    label="llama.cpp (local)",
    auth=LOCAL,
    detail="whatever llama-server is currently hosting",
    models=("local-model",),
    base_url="http://127.0.0.1:8080/v1",
  ),
)

PROVIDER_NAMES: tuple[str, ...] = tuple(provider.name for provider in PROVIDERS)
DEFAULT_PROVIDER = "chatgpt"


def find(name: str) -> Provider:
  """Look up one provider row by name, or explain which names exist."""
  for provider in PROVIDERS:
    if provider.name == name:
      return provider
  raise ValueError(f"unknown model provider {name!r}; choose one of {', '.join(PROVIDER_NAMES)}")


def by_auth(auth: str) -> tuple[Provider, ...]:
  """Return every provider that authenticates the given way, in menu order."""
  return tuple(provider for provider in PROVIDERS if provider.auth == auth)


# ---------------------------------------------------------------------------
# Stored credentials (~/.perferox/credentials.json)
# ---------------------------------------------------------------------------


def write_private(path: Path, text: str) -> None:
  """Write owner-only text through a temp file in the same directory."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.parent.chmod(0o700)
  # A fresh scratch name per write so two processes cannot interleave into one inode.
  handle, scratch = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
  try:
    with os.fdopen(handle, "w") as file:
      file.write(text)
    os.chmod(scratch, 0o600)
    os.replace(scratch, path)
  except BaseException:
    Path(scratch).unlink(missing_ok=True)
    raise


def read_credentials() -> dict[str, str]:
  """Read the stored credential map, treating a missing or broken file as empty."""
  try:
    loaded = json.loads(CREDENTIALS_PATH.read_text())
  except (OSError, ValueError):
    return {}
  return {str(key): str(value) for key, value in loaded.items()} if isinstance(loaded, dict) else {}


def save_credential(key: str, value: str) -> None:
  """Store one credential under `~/.perferox/credentials.json` with mode 0600."""
  credentials = read_credentials()
  credentials[key] = value
  write_private(CREDENTIALS_PATH, json.dumps(credentials, indent=2, sort_keys=True))


def api_key(provider: Provider) -> str:
  """Return the provider's API key from the environment, then from stored credentials."""
  from_env = os.environ.get(provider.key_env, "") if provider.key_env else ""
  return from_env or read_credentials().get(provider.name, "")


def account_id(provider: Provider) -> str:
  """Return the account id an account-scoped provider needs, if one is configured."""
  if not provider.account_env:
    return ""
  return os.environ.get(provider.account_env, "") or read_credentials().get(f"{provider.name}_account", "")


def base_url(provider: Provider) -> str:
  """Return the provider's endpoint with any account placeholder filled in."""
  if ACCOUNT_PLACEHOLDER not in provider.base_url:
    return provider.base_url
  resolved = account_id(provider)
  if not resolved:
    raise ValueError(f"{provider.label} needs an account id; set {provider.account_env} or re-run `perferox onboard`")
  return provider.base_url.replace(ACCOUNT_PLACEHOLDER, resolved)


# ---------------------------------------------------------------------------
# Stored profile (~/.perferox/config.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
  """The choices onboarding makes: which model to drive and which GPU cloud to rent."""

  provider: str
  model: str
  cloud: str = ""


def load_settings() -> Settings | None:
  """Read the saved profile, or None when the user has never onboarded."""
  try:
    loaded = json.loads(CONFIG_PATH.read_text())
  except (OSError, ValueError):
    return None
  if not isinstance(loaded, dict) or "provider" not in loaded:
    return None
  try:
    provider = find(str(loaded["provider"]))
  except ValueError:
    return None
  return Settings(provider=provider.name, model=str(loaded.get("model") or provider.default_model), cloud=str(loaded.get("cloud") or ""))


def save_settings(settings: Settings) -> None:
  """Persist the profile so detached workers resolve the same provider and model."""
  write_private(CONFIG_PATH, json.dumps({"provider": settings.provider, "model": settings.model, "cloud": settings.cloud}, indent=2))


def active_settings() -> Settings:
  """Resolve the effective profile: environment overrides beat the saved file."""
  settings = load_settings() or Settings(provider=DEFAULT_PROVIDER, model=find(DEFAULT_PROVIDER).default_model)
  provider = find(os.environ.get("PERFEROX_PROVIDER") or settings.provider)
  model = os.environ.get("PERFEROX_CHAT_MODEL") or (settings.model if provider.name == settings.provider else provider.default_model)
  return Settings(provider=provider.name, model=model, cloud=os.environ.get("PERFEROX_CLOUD") or settings.cloud)
