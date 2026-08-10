"""Sign in with an xAI account: OAuth 2.0 authorization code + PKCE.

`auth.x.ai` issues short-lived access tokens that the OpenAI-compatible API at
`https://api.x.ai/v1` accepts as plain bearer tokens, so the whole integration
is a credential source: log in once, store the refresh token, and hand a fresh
access token to every request. Tokens live in `~/.perferox/xai_oauth.json` with
mode 0600 and never touch the Perferox profile or a trace.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from time import time

import httpx

from perferox.providers import CONFIG_DIR, write_private

DISCOVERY_URL = "https://auth.x.ai/.well-known/openid-configuration"
# Public client id of the Grok CLI; a public OAuth client, so there is no secret.
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
# xAI only redirects to the loopback address registered for CLIENT_ID, so this
# port is fixed rather than preferred.
CALLBACK_PORT = 56121
CALLBACK_PATH = "/callback"
# Refresh the access token once it is this close to expiring.
EXPIRY_LEEWAY_S = 120
TOKEN_PATH = CONFIG_DIR / "xai_oauth.json"
DONE_PAGE = b"<html><body><h3>Perferox is signed in to xAI.</h3><p>You can close this tab.</p></body></html>"


@dataclass(frozen=True, slots=True)
class StoredTokens:
  """The persisted xAI session, including where to refresh it."""

  access_token: str
  refresh_token: str
  token_endpoint: str


def _validate_xai_https(url: str) -> None:
  """Require an HTTPS endpoint on x.ai before any credential is sent to it."""
  parsed = urllib.parse.urlparse(url)
  if parsed.scheme != "https":
    raise ValueError(f"endpoint {url} is not HTTPS; refusing to send credentials")
  host = parsed.hostname or ""
  if host != "x.ai" and not host.endswith(".x.ai"):
    raise ValueError(f"endpoint {url} is not on x.ai; refusing to send credentials")


def _pkce_pair() -> tuple[str, str]:
  """Return an RFC 7636 S256 verifier and its challenge."""
  verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")[:128]
  challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
  return verifier, challenge


def jwt_expiry(token: str) -> int | None:
  """Return a JWT's `exp` claim, or None when the token is not a readable JWT."""
  parts = token.split(".")
  if len(parts) < 2:
    return None
  payload = parts[1] + "=" * (-len(parts[1]) % 4)
  try:
    claims = json.loads(base64.urlsafe_b64decode(payload))
  except (ValueError, TypeError):
    return None
  expiry = claims.get("exp")
  return expiry if isinstance(expiry, int) else None


def expires_soon(token: str, now: float | None = None) -> bool:
  """Report whether a token is inside the refresh leeway; unreadable means live."""
  expiry = jwt_expiry(token)
  return expiry is not None and expiry <= (time() if now is None else now) + EXPIRY_LEEWAY_S


def load_tokens() -> StoredTokens | None:
  """Read the stored xAI session, or None when nobody has signed in."""
  try:
    stored = json.loads(TOKEN_PATH.read_text())
  except (OSError, ValueError):
    return None
  if not isinstance(stored, dict) or not stored.get("access_token"):
    return None
  return StoredTokens(str(stored["access_token"]), str(stored.get("refresh_token", "")), str(stored.get("token_endpoint", "")))


def save_tokens(tokens: StoredTokens) -> None:
  """Persist the xAI session owner-only, alongside the endpoint that refreshes it."""
  payload = {"access_token": tokens.access_token, "refresh_token": tokens.refresh_token, "token_endpoint": tokens.token_endpoint}
  write_private(TOKEN_PATH, json.dumps(payload, indent=2))


def clear_tokens() -> None:
  """Forget a revoked or expired session; a missing file is already forgotten."""
  TOKEN_PATH.unlink(missing_ok=True)


def _discover() -> tuple[str, str]:
  """Fetch xAI's OpenID configuration and return validated authorize/token endpoints."""
  response = httpx.get(DISCOVERY_URL, timeout=30.0)
  response.raise_for_status()
  document = response.json()
  authorize, token = str(document["authorization_endpoint"]), str(document["token_endpoint"])
  _validate_xai_https(authorize)
  _validate_xai_https(token)
  return authorize, token


def _authorize_url(authorize_endpoint: str, redirect_uri: str, challenge: str, state: str) -> str:
  """Build the browser URL that starts the sign-in."""
  query = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": redirect_uri,
    "scope": SCOPE,
    "code_challenge": challenge,
    "code_challenge_method": "S256",
    "state": state,
    "nonce": secrets.token_hex(16),
    # xAI rejects clients that do not name a plan.
    "plan": "generic",
    "referrer": "perferox",
  }
  separator = "&" if urllib.parse.urlparse(authorize_endpoint).query else "?"
  return f"{authorize_endpoint}{separator}{urllib.parse.urlencode(query)}"


def _exchange(token_endpoint: str, form: dict[str, str]) -> dict[str, object]:
  """POST one grant to the token endpoint and return the parsed response."""
  _validate_xai_https(token_endpoint)
  response = httpx.post(token_endpoint, data=form, headers={"Accept": "application/json"}, timeout=60.0)
  if response.status_code >= 400:
    raise RuntimeError(f"xAI token endpoint returned HTTP {response.status_code}: {response.text[:400]}")
  return response.json()


class _CallbackHandler(BaseHTTPRequestHandler):
  """Capture the single OAuth redirect and answer the browser with a done page."""

  def do_GET(self) -> None:
    """Record the code and state from the redirect, then close the page out."""
    parsed = urllib.parse.urlparse(self.path)
    if parsed.path != CALLBACK_PATH:
      self.send_error(404)
      return
    query = urllib.parse.parse_qs(parsed.query)
    # The server outlives this handler, so the result is parked on it.
    self.server.captured = {key: values[0] for key, values in query.items() if values}
    self.send_response(200)
    self.send_header("Content-Type", "text/html")
    self.send_header("Content-Length", str(len(DONE_PAGE)))
    self.end_headers()
    self.wfile.write(DONE_PAGE)

  def log_message(self, *_args: object) -> None:
    """Keep the throwaway callback server out of the terminal."""


def _await_callback(server: HTTPServer, timeout_s: float) -> dict[str, str]:
  """Serve exactly one redirect and return its query parameters."""
  server.timeout = timeout_s
  server.captured = {}
  server.handle_request()
  return server.captured


def _paste_callback(authorize_url: str) -> dict[str, str]:
  """Ask a headless session for the redirect URL its browser landed on."""
  print("Open this URL in a browser, finish the xAI sign-in, then paste the address bar back here:")
  print(f"\n  {authorize_url}\n")
  pasted = input("Redirect URL: ").strip()
  query = urllib.parse.urlparse(pasted).query or pasted
  return {key: values[0] for key, values in urllib.parse.parse_qs(query).items() if values}


def headless() -> bool:
  """Report whether this session can receive a loopback callback from a browser."""
  return bool(os.environ.get("SSH_CONNECTION")) or not sys.stdout.isatty()


def login(timeout_s: float = 300.0) -> StoredTokens:
  """Run the interactive xAI sign-in and persist the resulting session."""
  authorize_endpoint, token_endpoint = _discover()
  verifier, challenge = _pkce_pair()
  state = secrets.token_hex(16)
  redirect_uri = f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}"
  authorize_url = _authorize_url(authorize_endpoint, redirect_uri, challenge, state)

  if headless():
    callback = _paste_callback(authorize_url)
  else:
    # Bind before opening the browser so the redirect can never beat the listener.
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    try:
      webbrowser.open(authorize_url)
      print(f"Finish the xAI sign-in in your browser.\nIf it did not open: {authorize_url}")
      callback = _await_callback(server, timeout_s)
    finally:
      server.server_close()

  if callback.get("error"):
    raise RuntimeError(f"xAI sign-in failed: {callback['error']}")
  code = callback.get("code", "")
  if not code:
    raise RuntimeError("xAI sign-in did not return an authorization code")
  if callback.get("state") != state:
    raise RuntimeError("the xAI sign-in state did not match; start again")

  granted = _exchange(token_endpoint, {
    "grant_type": "authorization_code",
    "code": code,
    "redirect_uri": redirect_uri,
    "client_id": CLIENT_ID,
    "code_verifier": verifier,
    "code_challenge": challenge,
    "code_challenge_method": "S256",
  })
  tokens = StoredTokens(str(granted.get("access_token", "")), str(granted.get("refresh_token", "")), token_endpoint)
  if not tokens.access_token:
    raise RuntimeError("xAI returned no access token")
  save_tokens(tokens)
  return tokens


class XaiTokenProvider:
  """Hand out a live xAI access token, refreshing it just before it expires."""

  def __init__(self) -> None:
    """Start empty; the token file is read on first use, not at import."""
    self._tokens: StoredTokens | None = None
    # A refresh token is single use, so one refresh at a time or the loser is revoked.
    self._lock = threading.Lock()

  def get_token(self) -> str:
    """Return an access token that is valid now, refreshing under the lock if needed."""
    with self._lock:
      if self._tokens is None:
        self._tokens = load_tokens()
      if self._tokens is None:
        raise RuntimeError("not signed in to xAI; run `perferox login grok` first")
      if expires_soon(self._tokens.access_token):
        self._tokens = self._refresh(self._tokens)
      return self._tokens.access_token

  def _refresh(self, tokens: StoredTokens) -> StoredTokens:
    """Spend the refresh token for a new access token, or clear a dead session."""
    if not tokens.refresh_token or not tokens.token_endpoint:
      raise RuntimeError("the stored xAI session cannot be refreshed; run `perferox login grok` again")
    try:
      granted = _exchange(tokens.token_endpoint, {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": tokens.refresh_token})
    except RuntimeError:
      # The grant is gone for good; a retry would only spend a token that no longer exists.
      clear_tokens()
      raise
    refreshed = StoredTokens(
      str(granted.get("access_token", "")) or tokens.access_token,
      str(granted.get("refresh_token", "")) or tokens.refresh_token,
      tokens.token_endpoint,
    )
    save_tokens(refreshed)
    return refreshed


class BearerAuth(httpx.Auth):
  """Stamp every outgoing request with a freshly resolved bearer token."""

  def __init__(self, provider: XaiTokenProvider) -> None:
    """Hold the token provider that owns refresh and persistence."""
    self.provider = provider

  def auth_flow(self, request: httpx.Request):
    """Replace the Authorization header at send time so long runs never go stale."""
    request.headers["Authorization"] = f"Bearer {self.provider.get_token()}"
    yield request


def auth_ready() -> bool:
  """Report whether a usable, refreshable xAI session is on disk."""
  try:
    XaiTokenProvider().get_token()
  except Exception:  # noqa: BLE001 - any failure means "sign in again"
    return False
  return True


def token_file() -> Path:
  """Return the path the xAI session is stored at, for diagnostics."""
  return TOKEN_PATH
