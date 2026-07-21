"""ChatGPT OAuth auth and model construction for Perferox."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


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


def build_chat_model(model: str | None = None):
  """Build Perferox's OAuth-backed LangChain chat model."""
  from langchain_openai.chat_models.codex import _ChatOpenAICodex
  provider = _chatgpt_provider()
  model_name = model or os.environ.get("PERFEROX_CHAT_MODEL", "gpt-5.5")
  return _ChatOpenAICodex(
    model=model_name,
    originator="perferox",
    token_provider=provider,
  )
