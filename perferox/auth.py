"""ChatGPT OAuth auth and model construction for Perferox."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def cloud_provider(api_key: str) -> str:
  """Infer the provider from its API-key prefix."""
  if api_key.startswith("secret_"): return "lambda"
  if api_key.startswith("rpa_"): return "runpod"
  raise ValueError("API key must start with secret_ for Lambda or rpa_ for RunPod")


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


def build_chat_model(model: str | None = None, *, role: str = "main"):
  """Build the role-tuned OAuth-backed LangChain chat model."""
  from langchain_openai.chat_models.codex import _ChatOpenAICodex
  provider = _chatgpt_provider()
  if role not in {"main", "subagent"}:
    raise ValueError(f"unknown model role: {role}")
  is_subagent = role == "subagent"
  role_model = os.environ.get("PERFEROX_SUBAGENT_MODEL") if is_subagent else None
  model_name = model or role_model or os.environ.get("PERFEROX_CHAT_MODEL", "gpt-5.5")
  return _ChatOpenAICodex(
    model=model_name,
    originator="perferox",
    reasoning_effort="medium" if is_subagent else "high",
    token_provider=provider,
  )
