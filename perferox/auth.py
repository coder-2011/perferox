"""ChatGPT OAuth auth and model construction for Perferox."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_CHAT_MODEL = "gpt-5.5"
MODEL_ENV_VAR = "PERFEROX_CHAT_MODEL"
ORIGINATOR = "perferox"


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


def chatgpt_auth_ready() -> bool:
  """Return whether a refreshable ChatGPT OAuth token is available."""
  try:
    from langchain_openai.chatgpt_oauth import _FileChatGPTOAuthTokenProvider

    _FileChatGPTOAuthTokenProvider.from_default_store().get_token()
  except Exception:  # noqa: BLE001
    return False
  return True


def login_chatgpt_oauth(timeout_s: float = 300.0) -> None:
  """Run the browser-based ChatGPT OAuth login flow."""
  from langchain_openai.chatgpt_oauth import login_chatgpt

  login_chatgpt(timeout=timeout_s, open_browser=True)


def build_chat_model(model: str | None = None) -> BaseChatModel:
  """Build Perferox's OAuth-backed LangChain chat model."""
  from langchain_openai.chat_models.codex import _ChatOpenAICodex
  from langchain_openai.chatgpt_oauth import _FileChatGPTOAuthTokenProvider

  provider = _FileChatGPTOAuthTokenProvider.from_default_store()
  provider.get_token()
  model_name = model or os.environ.get(MODEL_ENV_VAR, DEFAULT_CHAT_MODEL)
  return _ChatOpenAICodex(
    model=model_name,
    originator=ORIGINATOR,
    token_provider=provider,
  )
