"""LLM OAuth, model selection, and cloud credentials."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_litellm import ChatLiteLLM

OAUTH_MODELS = {
  "chatgpt": "chatgpt/gpt-5.4",
  "github-copilot": "github_copilot/gpt-4",
}


def perferox_auth_probe() -> None:
  """Confirm that the selected model can call Perferox tools."""


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


def _provider_path() -> Path:
  """Return the XDG-style path for Perferox's selected provider."""
  config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
  return config_root / "perferox" / "provider"


def active_model() -> str | None:
  """Load the default model for the selected OAuth provider."""
  try:
    provider = _provider_path().read_text(encoding="utf-8").strip()
  except (OSError, UnicodeError):
    return None
  return OAUTH_MODELS.get(provider)


def login_provider(provider: str) -> str:
  """Run CLI-owned OAuth and save the provider after a real tool call."""
  model = OAUTH_MODELS.get(provider)
  if model is None:
    raise ValueError(f"unsupported OAuth provider: {provider}")
  chat_model = ChatLiteLLM(model=model, max_retries=0)
  probe_model = chat_model.bind_tools([perferox_auth_probe], tool_choice="required")
  response = probe_model.invoke(f"Call {perferox_auth_probe.__name__} once with no arguments.")
  if not response.tool_calls:
    raise RuntimeError(f"{model} authenticated but did not complete the tool-call probe")
  provider_path = _provider_path()
  provider_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  provider_path.write_text(provider, encoding="utf-8")
  return model


def build_chat_model():
  """Build the active OAuth-backed LangChain model without starting login."""
  model = active_model()
  if model is None:
    raise RuntimeError("LLM OAuth is missing; run `perferox login` first")

  return ChatLiteLLM(model=model, max_retries=0)
