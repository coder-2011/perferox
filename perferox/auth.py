"""LLM OAuth, model selection, and cloud credentials."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

LEGACY_MODELS = {
  "chatgpt": "chatgpt/gpt-5.4",
  "github-copilot": "github_copilot/gpt-4",
}


@dataclass(frozen=True, slots=True)
class ModelProfile:
  """Store one non-secret model selection."""

  model: str
  reasoning_effort: str | None = None

  def __post_init__(self) -> None:
    """Normalize and validate values before they reach a provider."""
    model = self.model.strip()
    effort = self.reasoning_effort.strip() if self.reasoning_effort else None
    if not model:
      raise ValueError("model must be a non-empty model name")
    object.__setattr__(self, "model", model)
    object.__setattr__(self, "reasoning_effort", effort)

  def label(self) -> str:
    """Render the model and optional reasoning effort compactly."""
    return f"{self.model} · {self.reasoning_effort}" if self.reasoning_effort else self.model


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


def _profile_path() -> Path:
  """Return the XDG-style path for Perferox's selected model profile."""
  config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
  return config_root / "perferox" / "model.json"


def active_model_profile() -> ModelProfile | None:
  """Load the selected model profile, including legacy provider selections."""
  profile_path = _profile_path()
  try:
    profile_text = profile_path.read_text(encoding="utf-8")
  except OSError:
    profile_text = ""
  except UnicodeError:
    return None
  if profile_text:
    try:
      data = json.loads(profile_text)
      return ModelProfile(model=data["model"], reasoning_effort=data.get("reasoning_effort"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
      return None
  try:
    provider = profile_path.with_name("provider").read_text(encoding="utf-8").strip()
  except (OSError, UnicodeError):
    return None
  model = LEGACY_MODELS.get(provider)
  return ModelProfile(model) if model else None


def login_model(model: str, reasoning_effort: str | None = None) -> ModelProfile:
  """Validate and atomically save one supported model profile."""
  profile = ModelProfile(model, reasoning_effort)
  if profile.model.startswith("chatgpt/"):
    from langchain_openai.chatgpt_oauth import _ChatGPTOAuthRefreshError, _FileChatGPTOAuthTokenProvider, login_chatgpt

    try:
      _FileChatGPTOAuthTokenProvider.from_default_store().get_token()
    except (FileNotFoundError, _ChatGPTOAuthRefreshError):
      login_chatgpt()
  chat_model = build_chat_model(profile, max_retries=0)
  probe_model = chat_model.bind_tools([perferox_auth_probe], tool_choice="required")
  response = probe_model.invoke(f"Call {perferox_auth_probe.__name__} once with no arguments.")
  if not response.tool_calls:
    raise RuntimeError(f"{profile.model} authenticated but did not complete the tool-call probe")
  profile_path = _profile_path()
  profile_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  temporary_path = profile_path.with_suffix(".tmp")
  temporary_path.write_text(json.dumps({"model": profile.model, "reasoning_effort": profile.reasoning_effort}, separators=(",", ":")), encoding="utf-8")
  temporary_path.replace(profile_path)
  return profile


def build_chat_model(profile: ModelProfile, *, max_retries: int = 1):
  """Build ChatGPT models with LangChain OpenAI and others with LiteLLM."""
  if profile.model.startswith("chatgpt/"):
    from langchain_openai.chat_models.codex import _ChatOpenAICodex
    from langchain_openai.chatgpt_oauth import _FileChatGPTOAuthTokenProvider

    return _ChatOpenAICodex(
      model=profile.model.removeprefix("chatgpt/"),
      reasoning_effort=profile.reasoning_effort,
      originator="perferox",
      token_provider=_FileChatGPTOAuthTokenProvider.from_default_store(),
      max_retries=max_retries,
    )
  from langchain_litellm import ChatLiteLLM

  model_kwargs = {"reasoning_effort": profile.reasoning_effort} if profile.reasoning_effort else {}
  return ChatLiteLLM(model=profile.model, model_kwargs=model_kwargs, max_retries=max_retries)
