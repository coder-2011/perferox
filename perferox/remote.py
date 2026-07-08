"""Paramiko remote sessions kept outside LangGraph state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


READ_CHUNK_BYTES = 32768
MAX_DRAIN_READS = 16
POST_EXIT_DRAIN_S = 0.2


@dataclass(frozen=True, slots=True)
class RemoteResult:
  """Captured output from one remote command."""

  command: str
  exit_status: int
  stdout: str
  stderr: str


class RemoteCommandTimeout(TimeoutError):
  """Timeout raised with partial remote command output attached."""

  def __init__(self, command: str, timeout_s: float, stdout: str, stderr: str):
    """Store partial output for setup/benchmark failure logging."""
    super().__init__(f"remote command timed out after {timeout_s}s: {command}")
    self.command = command
    self.timeout_s = timeout_s
    self.stdout = stdout
    self.stderr = stderr


class RemoteSession:
  """Own one SSH connection and expose small file/command operations."""

  def __init__(
    self,
    *,
    session_id: str,
    host: str,
    user: str = "root",
    port: int = 22,
    key_filename: str | None = None,
  ):
    """Store connection metadata without opening the socket."""
    self.session_id = session_id
    self.host = host
    self.user = user
    self.port = port
    self.key_filename = key_filename
    self._client: paramiko.SSHClient | None = None

  @classmethod
  def connect(
    cls,
    *,
    session_id: str,
    host: str,
    user: str = "root",
    port: int = 22,
    key_filename: str | None = None,
    password: str | None = None,
    trust_unknown_host: bool = False,
    allow_agent: bool = False,
    look_for_keys: bool = False,
    timeout_s: float = 30.0,
  ) -> RemoteSession:
    """Open an SSH connection and return the live session wrapper."""
    session = cls(
      session_id=session_id,
      host=host,
      user=user,
      port=port,
      key_filename=key_filename,
    )
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    policy = paramiko.AutoAddPolicy() if trust_unknown_host else paramiko.RejectPolicy()
    client.set_missing_host_key_policy(policy)
    try:
      client.connect(
        hostname=host,
        port=port,
        username=user,
        key_filename=key_filename,
        password=password,
        timeout=timeout_s,
        banner_timeout=timeout_s,
        auth_timeout=timeout_s,
        allow_agent=allow_agent,
        look_for_keys=look_for_keys,
      )
    except Exception:
      client.close()
      raise
    session._client = client
    return session

  def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
    """Run a remote shell command and collect stdout/stderr without deadlock."""
    transport = self._client_or_raise().get_transport()
    if transport is None or not transport.is_active():
      raise RuntimeError(f"remote session is not connected: {self.session_id}")

    stdout: list[bytes] = []
    stderr: list[bytes] = []
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    channel = transport.open_session(timeout=_remaining_s(deadline))
    try:
      channel.settimeout(_remaining_s(deadline))
      channel.exec_command(command)
      channel.shutdown_write()
      exit_status = None
      post_exit_deadline = None
      while True:
        _raise_if_timed_out(deadline, command, timeout_s, stdout, stderr)
        # Drain both streams while polling so large output cannot block exit status.
        drained = _drain_channel(channel, stdout, stderr)
        if exit_status is None and channel.exit_status_ready():
          exit_status = channel.recv_exit_status()
          post_exit_deadline = time.monotonic() + POST_EXIT_DRAIN_S
        if exit_status is not None and drained:
          post_exit_deadline = time.monotonic() + POST_EXIT_DRAIN_S
        if exit_status is None and channel.closed and not _output_ready(channel):
          exit_status = channel.recv_exit_status()
          break
        if exit_status is not None and channel.closed and not _output_ready(channel):
          break
        if exit_status is not None and post_exit_deadline is not None:
          if time.monotonic() >= post_exit_deadline:
            break
        time.sleep(0.05)
    finally:
      channel.close()

    return RemoteResult(
      command=command,
      exit_status=exit_status,
      stdout=_decode(stdout),
      stderr=_decode(stderr),
    )

  def put(self, local_path: str | Path, remote_path: str) -> None:
    """Copy a local file to the remote host over SFTP."""
    sftp = self._client_or_raise().open_sftp()
    try:
      sftp.put(str(local_path), remote_path)
    finally:
      sftp.close()

  def close(self) -> None:
    """Close the SSH client if it is open."""
    client = self._client
    self._client = None
    if client is not None:
      client.close()

  def _client_or_raise(self) -> paramiko.SSHClient:
    """Return the live Paramiko client or fail before a remote operation."""
    if self._client is None:
      raise RuntimeError(f"remote session is not connected: {self.session_id}")
    return self._client


class SessionRegistry:
  """Keep live SSH clients behind string ids for graph/tool lookup."""

  def __init__(self):
    """Create an empty in-memory session registry."""
    self._sessions: dict[str, RemoteSession] = {}

  def add(self, session: RemoteSession) -> str:
    """Register a live session and return its graph-safe id."""
    if session.session_id in self._sessions:
      raise ValueError(f"remote session already exists: {session.session_id}")
    self._sessions[session.session_id] = session
    return session.session_id

  def get(self, session_id: str) -> RemoteSession:
    """Look up a live session by the id stored in graph state."""
    try:
      return self._sessions[session_id]
    except KeyError:
      raise KeyError(f"unknown remote session: {session_id}") from None

  def close(self, session_id: str) -> None:
    """Close and forget one registered session."""
    session = self.get(session_id)
    try:
      session.close()
    finally:
      del self._sessions[session_id]

  def close_all(self) -> None:
    """Close every registered session during host shutdown."""
    first_error = None
    for session_id in list(self._sessions):
      try:
        self.close(session_id)
      except Exception as exc:
        first_error = first_error or exc
    if first_error is not None:
      raise RuntimeError("failed to close one or more remote sessions") from first_error


def _drain_channel(channel: paramiko.Channel, stdout: list[bytes], stderr: list[bytes]) -> bool:
  """Move available SSH channel bytes into caller-owned buffers."""
  drained = False
  for _ in range(MAX_DRAIN_READS):
    ready = False
    if channel.recv_ready():
      stdout.append(channel.recv(READ_CHUNK_BYTES))
      ready = True
      drained = True
    if channel.recv_stderr_ready():
      stderr.append(channel.recv_stderr(READ_CHUNK_BYTES))
      ready = True
      drained = True
    if not ready:
      break
  return drained


def _output_ready(channel: paramiko.Channel) -> bool:
  """Return whether stdout or stderr has buffered bytes ready to drain."""
  return channel.recv_ready() or channel.recv_stderr_ready()


def _remaining_s(deadline: float | None) -> float | None:
  """Return the remaining seconds before a monotonic deadline."""
  if deadline is None:
    return None
  return max(deadline - time.monotonic(), 0.0)


def _raise_if_timed_out(
  deadline: float | None,
  command: str,
  timeout_s: float | None,
  stdout: list[bytes],
  stderr: list[bytes],
) -> None:
  """Raise a timeout that preserves remote output collected so far."""
  if deadline is not None and time.monotonic() >= deadline:
    raise RemoteCommandTimeout(command, timeout_s or 0.0, _decode(stdout), _decode(stderr))


def _decode(chunks: list[bytes]) -> str:
  """Decode remote output bytes while preserving invalid data as text."""
  return b"".join(chunks).decode(errors="replace")
