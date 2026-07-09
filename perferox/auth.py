"""ChatGPT OAuth auth and model construction for Perferox."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_CHAT_MODEL = "gpt-5.5"
MODEL_ENV_VAR = "PERFEROX_CHAT_MODEL"
ORIGINATOR = "perferox"


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
