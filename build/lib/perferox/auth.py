"""LLM OAuth profiles, model construction, and cloud credentials."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_litellm import ChatLiteLLM


@dataclass(frozen=True, slots=True)
class OAuthProvider:
  """Describe one CLI-supported LiteLLM OAuth provider."""

  label: str
  model_prefix: str
  default_model: str
  token_dir_env: str
  default_token_dir: str
  token_file_env: str
  default_token_file: str


@dataclass(frozen=True, slots=True)
class ActiveModel:
  """Pair the selected model with its derived OAuth provider."""

  provider: str
  model: str


OAUTH_PROVIDERS = {
  "chatgpt": OAuthProvider(
    label="ChatGPT subscription",
    model_prefix="chatgpt/",
    default_model="chatgpt/gpt-5.4",
    token_dir_env="CHATGPT_TOKEN_DIR",
    default_token_dir="~/.config/litellm/chatgpt",
    token_file_env="CHATGPT_AUTH_FILE",
    default_token_file="auth.json",
  ),
  "github-copilot": OAuthProvider(
    label="GitHub Copilot",
    model_prefix="github_copilot/",
    default_model="github_copilot/gpt-4",
    token_dir_env="GITHUB_COPILOT_TOKEN_DIR",
    default_token_dir="~/.config/litellm/github_copilot",
    token_file_env="GITHUB_COPILOT_ACCESS_TOKEN_FILE",
    default_token_file="access-token",
  ),
}

AUTH_PROBE_TOOL = {
  "type": "function",
  "function": {
    "name": "perferox_auth_probe",
    "description": "Confirm that the selected model can call Perferox tools.",
    "parameters": {"type": "object", "properties": {}},
  },
}


class AuthenticatedChatLiteLLM(ChatLiteLLM):
  """Refuse model calls that would need an interactive OAuth refresh."""

  def invoke(self, input: Any, config: Any = None, *, stop: list[str] | None = None, **kwargs: Any) -> Any:
    """Recheck local auth before every synchronous LangGraph model call."""
    if active_model() is None:
      raise RuntimeError("LLM OAuth expired; run `perferox login` again")
    return super().invoke(input, config, stop=stop, **kwargs)


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


def _profile_path() -> Path:
  """Return the XDG-style path for Perferox's non-secret LLM profile."""
  config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
  return config_root / "perferox" / "llm.json"


def _token_path(provider: OAuthProvider) -> Path:
  """Return the LiteLLM-owned credential path for one provider."""
  token_dir = Path(os.environ.get(provider.token_dir_env, provider.default_token_dir)).expanduser()
  token_file = os.environ.get(provider.token_file_env, provider.default_token_file)
  return token_dir / token_file


def active_model() -> ActiveModel | None:
  """Load the selected model only when its local OAuth is usable."""
  try:
    data = json.loads(_profile_path().read_text(encoding="utf-8"))
    model = data["model"]
  except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
    return None
  if not isinstance(model, str):
    return None
  for provider_name, provider in OAUTH_PROVIDERS.items():
    if model.startswith(provider.model_prefix) and _credentials_ready(provider_name):
      return ActiveModel(provider=provider_name, model=model)
  return None


def _credentials_ready(provider_name: str) -> bool:
  """Check one provider's LiteLLM-owned token without starting OAuth."""
  token_path = _token_path(OAUTH_PROVIDERS[provider_name])
  try:
    if provider_name == "chatgpt":
      auth_data = json.loads(token_path.read_text(encoding="utf-8"))
      expires_at = auth_data.get("expires_at")
      credentials_present = bool(auth_data.get("access_token") and auth_data.get("refresh_token"))
      return credentials_present and isinstance(expires_at, (int, float)) and expires_at > time.time() + 60
    return bool(token_path.read_text(encoding="utf-8").strip())
  except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
    return False


def _save_active_model(active: ActiveModel) -> None:
  """Atomically save a validated model selection."""
  profile_path = _profile_path()
  profile_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile("w", dir=profile_path.parent, encoding="utf-8", delete=False) as file:
      json.dump({"model": active.model}, file, separators=(",", ":"))
      temporary_path = Path(file.name)
    temporary_path.replace(profile_path)
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)


def login_model(provider_name: str, model: str | None = None) -> ActiveModel:
  """Run CLI-owned OAuth and save the model only after a real tool call."""
  try:
    provider = OAUTH_PROVIDERS[provider_name]
  except KeyError as exc:
    raise ValueError(f"unsupported OAuth provider: {provider_name}") from exc
  model_name = model or provider.default_model
  if not model_name.startswith(provider.model_prefix):
    raise ValueError(f"{provider.label} models must start with {provider.model_prefix}")

  chat_model = ChatLiteLLM(model=model_name, max_retries=0)
  probe_model = chat_model.bind_tools([AUTH_PROBE_TOOL], tool_choice="required")
  response = probe_model.invoke("Call perferox_auth_probe once with no arguments.")
  if not any(call.get("name") == "perferox_auth_probe" for call in response.tool_calls):
    raise RuntimeError(f"{model_name} authenticated but did not complete the tool-call probe")
  if not _credentials_ready(provider_name):
    raise RuntimeError(f"{provider.label} did not persist usable OAuth credentials")
  active = ActiveModel(provider=provider_name, model=model_name)
  _save_active_model(active)
  return active


def build_chat_model():
  """Build the active OAuth-backed LangChain model without starting login."""
  active = active_model()
  if active is None:
    raise RuntimeError("LLM OAuth is missing; run `perferox login` first")

  return AuthenticatedChatLiteLLM(model=active.model, max_retries=0)
