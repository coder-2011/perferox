"""Model-provider auth and chat-model construction for Perferox.

Two unrelated credentials meet here. The *model* credential selects and proves
a chat backend (a ChatGPT or xAI account sign-in, an API key, or a local server
that needs neither). The *cloud* credential rents the GPU a benchmark runs on.
They are kept apart on purpose: workers receive only the cloud key for the
provider they were launched with, and read the model credential from the
Perferox profile.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from perferox import providers
from perferox.providers import KEY, LOCAL, OAUTH, Provider

# Cloud GPU providers a benchmark worker can rent an instance from.
CLOUD_PROVIDERS = ("runpod", "lambda", "modal")


def cloud_provider(api_key: str) -> str:
  """Identify the provider from a key prefix or explicit Modal selection."""
  if api_key.partition("\n")[0] == "modal": return "modal"
  if api_key.startswith("secret_"): return "lambda"
  if api_key.startswith("rpa_"): return "runpod"
  raise ValueError("API key must start with secret_ for Lambda or rpa_ for RunPod; Modal uses `modal setup`")


def modal_cloud_key() -> str:
  """Build the one-use Modal handoff from paired environment tokens or a local profile."""
  token_id = os.environ.get("MODAL_TOKEN_ID", "")
  token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
  if not token_id and not token_secret:
    if not Path("~/.modal.toml").expanduser().is_file():
      raise ValueError("Modal auth is missing; run `modal setup` or set both Modal token variables")
    return "modal"
  if not token_id or not token_secret:
    raise ValueError("Modal requires both MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
  if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
    raise ValueError("Modal API tokens must start with ak- and as-")
  return f"modal\n{token_id}\n{token_secret}"


def cloud_environment(api_key: str) -> dict[str, str]:
  """Return only the selected provider credentials for a worker process."""
  provider = cloud_provider(api_key)
  if provider == "lambda":
    return {"LAMBDA_API_KEY": api_key}
  if provider == "runpod":
    return {"RUNPOD_API_KEY": api_key}
  parts = api_key.splitlines()
  if parts == ["modal"]:
    return {}
  if len(parts) != 3 or not parts[1].startswith("ak-") or not parts[2].startswith("as-"):
    raise ValueError("invalid Modal credential handoff")
  return {"MODAL_TOKEN_ID": parts[1], "MODAL_TOKEN_SECRET": parts[2]}


def write_cloud_key(api_key: str) -> Path:
  """Write a mode-0600 key handoff for one detached agent process."""
  with tempfile.NamedTemporaryFile("w", prefix="perferox-key-", delete=False) as file:
    file.write(api_key)
    return Path(file.name)


def read_cloud_key(path: str | Path) -> str:
  """Read and delete a one-use API-key handoff."""
  key_path = Path(path)
  try:
    return key_path.read_text()
  finally:
    key_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ChatGPT account sign-in
# ---------------------------------------------------------------------------


def _chatgpt_provider():
  """Open the persisted provider and require a usable ChatGPT token."""
  from langchain_openai.chatgpt_oauth import _FileChatGPTOAuthTokenProvider

  provider = _FileChatGPTOAuthTokenProvider.from_default_store()
  provider.get_token()
  return provider


def chatgpt_auth_ready() -> bool:
  """Return whether a refreshable ChatGPT OAuth token is available."""
  try:
    _chatgpt_provider()
  except Exception:  # noqa: BLE001
    return False
  return True


def ensure_chatgpt_auth(timeout_s: float = 300.0) -> bool:
  """Ensure persisted ChatGPT OAuth, returning whether login was needed."""
  if chatgpt_auth_ready():
    return False
  from langchain_openai.chatgpt_oauth import login_chatgpt, login_chatgpt_device

  # SSH and non-interactive sessions cannot reliably receive a loopback callback.
  headless = os.environ.get("SSH_CONNECTION") or not sys.stdout.isatty()
  login = login_chatgpt_device if headless else login_chatgpt
  provider = login(timeout=timeout_s)
  provider.get_token()
  return True


# ---------------------------------------------------------------------------
# Provider-agnostic model auth
# ---------------------------------------------------------------------------


def auth_ready(provider: Provider) -> bool:
  """Report whether this provider could be used right now without any prompting."""
  if provider.auth == LOCAL:
    return True
  if provider.auth == KEY:
    if provider.account_env and not providers.account_id(provider):
      return False
    return bool(providers.api_key(provider))
  if provider.name == "chatgpt":
    return chatgpt_auth_ready()
  from perferox import xai_oauth

  return xai_oauth.auth_ready()


def ensure_auth(provider: Provider, timeout_s: float = 300.0) -> bool:
  """Bring a provider to a usable state, returning whether a sign-in happened."""
  if auth_ready(provider):
    return False
  if provider.auth != OAUTH:
    raise ValueError(missing_credential(provider))
  if provider.name == "chatgpt":
    return ensure_chatgpt_auth(timeout_s)
  from perferox import xai_oauth

  xai_oauth.login(timeout_s)
  return True


def missing_credential(provider: Provider) -> str:
  """Explain, in one line, what this provider still needs."""
  if provider.auth == OAUTH:
    return f"{provider.label} is not signed in; run `perferox login {provider.name}`"
  if provider.account_env and not providers.account_id(provider):
    return f"{provider.label} needs an account id; set {provider.account_env} or run `perferox onboard`"
  return f"{provider.label} needs an API key; set {provider.key_env} or run `perferox onboard`"


# ---------------------------------------------------------------------------
# Chat model construction
# ---------------------------------------------------------------------------


def _chatgpt_model(model: str):
  """Build the ChatGPT-subscription chat model over the Codex Responses API."""
  from langchain_openai.chat_models.codex import _ChatOpenAICodex

  return _ChatOpenAICodex(model=model, originator="perferox", token_provider=_chatgpt_provider())


def _grok_model(provider: Provider, model: str):
  """Build a Grok chat model whose bearer token is refreshed per request."""
  import httpx
  from langchain_openai import ChatOpenAI

  from perferox.xai_oauth import BearerAuth, XaiTokenProvider

  tokens = XaiTokenProvider()
  auth = BearerAuth(tokens)
  return ChatOpenAI(
    model=model,
    base_url=provider.base_url,
    api_key=tokens.get_token(),
    http_client=httpx.Client(auth=auth, timeout=600.0),
    http_async_client=httpx.AsyncClient(auth=auth, timeout=600.0),
  )


def _compatible_model(provider: Provider, model: str):
  """Build a chat model for any provider that speaks OpenAI chat completions."""
  from langchain_openai import ChatOpenAI

  key = providers.api_key(provider)
  if provider.auth == KEY and not key:
    raise ValueError(missing_credential(provider))
  headers = {"HTTP-Referer": "https://github.com/coder-2011/perferox", "X-Title": "Perferox"} if provider.name == "openrouter" else None
  # Local servers ignore the key but the client still requires a non-empty one.
  return ChatOpenAI(model=model, base_url=providers.base_url(provider), api_key=key or "local", default_headers=headers)


def build_chat_model(model: str | None = None, provider_name: str | None = None):
  """Build the configured chat model, or an explicitly requested one."""
  settings = providers.active_settings()
  provider = providers.find(provider_name or settings.provider)
  model_name = model or (settings.model if provider.name == settings.provider else provider.default_model)
  if provider.name == "chatgpt":
    return _chatgpt_model(model_name)
  if provider.name == "grok":
    return _grok_model(provider, model_name)
  return _compatible_model(provider, model_name)
