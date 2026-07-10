from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field

import paramiko

READ_CHUNK_BYTES = 32768
MAX_DRAIN_READS = 16
REMOTE_KILL_GRACE_S = 5.0


@dataclass(slots=True)
class RemoteResult:
    """Captured output from one remote command."""

    exit_status: int | None
    stdout: str
    stderr: str


@dataclass(eq=False, slots=True)
class RemoteSession:
    """Own one SSH connection and expose small file/command operations."""

    session_id: str
    _client: paramiko.SSHClient | None = field(default=None, init=False, repr=False)

    @classmethod
    def connect(cls, session_id: str, host: str, user: str = "root", port: int = 22, timeout_s: float = 30.0) -> RemoteSession:
        """Connect one host-owned session id to an SSH server."""
        session = cls(session_id)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, port=port, username=user, timeout=timeout_s, banner_timeout=timeout_s, auth_timeout=timeout_s)
        except Exception as exc:
            client.close()
            raise RuntimeError(f"failed to connect to {user}@{host}:{port}") from exc
        session._client = client
        return session

    def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
        """Run a command and terminate its remote process group on timeout."""
        client = self._client
        transport = None if client is None else client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError(f"remote session is not connected: {self.session_id}")

        stdout, stderr = bytearray(), bytearray()
        monotonic = time.monotonic
        deadline = None if timeout_s is None else monotonic() + timeout_s + REMOTE_KILL_GRACE_S + 1.0
        channel = transport.open_session(timeout=min(timeout_s or 30.0, 30.0))

        command = f"bash -lc {shlex.quote(command)}"
        if timeout_s is not None:
            # GNU timeout owns a new process group and escalates if TERM is ignored.
            command = (
                f"timeout --signal=TERM --kill-after={REMOTE_KILL_GRACE_S}s "
                f"{timeout_s}s {command}"
            )

        def drain() -> bool:
            """Drain bounded output chunks without starving timeout checks."""
            drained = False
            for _ in range(MAX_DRAIN_READS):
                stdout_ready = channel.recv_ready()
                stderr_ready = channel.recv_stderr_ready()
                if not stdout_ready and not stderr_ready:
                    break
                drained = True
                if stdout_ready:
                    stdout.extend(channel.recv(READ_CHUNK_BYTES))
                if stderr_ready:
                    stderr.extend(channel.recv_stderr(READ_CHUNK_BYTES))
            return drained

        def result(exit_status: int | None) -> RemoteResult:
            """Decode captured output into one command result."""
            return RemoteResult(exit_status, stdout.decode(errors="replace"), stderr.decode(errors="replace"))

        try:
            channel.settimeout(timeout_s)
            channel.exec_command(command)
            channel.shutdown_write()
            while not channel.exit_status_ready():
                drained = drain()
                if deadline is not None and monotonic() >= deadline:
                    return result(None)
                if not drained:
                    time.sleep(0.05)
            while drain():
                pass
            return result(channel.recv_exit_status())
        finally:
            channel.close()

    def close(self) -> None:
        """Disconnect this session once and clear its client."""
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    def is_connected(self) -> bool:
        """Return whether the SSH transport is currently usable."""
        client = self._client
        transport = None if client is None else client.get_transport()
        return transport is not None and transport.is_active()


class SessionRegistry:
    """Keep live SSH clients behind string ids for graph/tool lookup."""

    def __init__(self) -> None:
        """Create an empty host-owned session map."""
        self._sessions: dict[str, RemoteSession] = {}

    def add(self, session: RemoteSession) -> None:
        """Register a session without replacing an existing id."""
        if session.session_id in self._sessions:
            raise ValueError(f"remote session already exists: {session.session_id}")
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> RemoteSession:
        """Return the session registered under an id."""
        return self._sessions[session_id]

    def connected(self, session_id: str) -> bool:
        """Return whether an id currently owns a live SSH transport."""
        session = self._sessions.get(session_id)
        return session is not None and session.is_connected()

    def close(self, session_id: str) -> None:
        """Remove and disconnect an id when it exists."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        """Disconnect and forget every registered session."""
        while self._sessions:
            self._sessions.popitem()[1].close()
