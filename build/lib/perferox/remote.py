from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import paramiko

READ_CHUNK_BYTES = 32768
MAX_DRAIN_READS = 16


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
        """Run a command, returning partial output and no status on timeout."""
        client = self._client
        transport = None if client is None else client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError(f"remote session is not connected: {self.session_id}")

        stdout, stderr = bytearray(), bytearray()
        monotonic = time.monotonic
        deadline = None if timeout_s is None else monotonic() + timeout_s
        channel = transport.open_session(timeout=timeout_s)

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


@dataclass(eq=False, slots=True)
class ModalSession:
    """Run commands through one Modal Sandbox handle."""

    session_id: str
    _sandbox: Any | None = field(repr=False)

    def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
        """Execute one command through Modal's native Sandbox process API."""
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError(f"remote session is not connected: {self.session_id}")
        argv = shlex.split(command)
        if timeout_s is None:
            process = sandbox.exec(*argv)
        else:
            process = sandbox.exec(*argv, timeout=max(1, ceil(timeout_s)))
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        exit_status = process.wait()
        return RemoteResult(exit_status, stdout, stderr)

    def close(self) -> None:
        """Detach the client while host-owned cleanup terminates the Sandbox."""
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is not None:
            # Host-owned cleanup reattaches by id, so client detachment is best effort.
            try:
                sandbox.detach()
            except Exception:  # noqa: BLE001,S110
                pass


class SessionRegistry:
    """Keep live remote execution sessions behind string ids for tool lookup."""

    def __init__(self) -> None:
        """Create an empty host-owned session map."""
        self._sessions: dict[str, RemoteSession | ModalSession] = {}

    def add(self, session: RemoteSession | ModalSession) -> None:
        """Register a session without replacing an existing id."""
        if session.session_id in self._sessions:
            raise ValueError(f"remote session already exists: {session.session_id}")
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> RemoteSession | ModalSession:
        """Return the session registered under an id."""
        return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        """Remove and disconnect an id when it exists."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        """Disconnect and forget every registered session."""
        while self._sessions:
            self._sessions.popitem()[1].close()
