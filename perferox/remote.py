from __future__ import annotations

import time
from dataclasses import dataclass, field

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
    def connect(
        cls,
        session_id: str,
        host: str,
        user: str = "root",
        port: int = 22,
        timeout_s: float = 30.0,
    ) -> RemoteSession:
        session = cls(session_id=session_id)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=user,
                timeout=timeout_s,
                banner_timeout=timeout_s,
                auth_timeout=timeout_s,
            )
        except Exception as exc:
            client.close()
            raise RuntimeError(f"failed to connect to {user}@{host}:{port}") from exc
        session._client = client
        return session

    def run(self, command: str, *, timeout_s: float | None = None) -> RemoteResult:
        connection_error = f"remote session is not connected: {self.session_id}"
        if self._client is None:
            raise RuntimeError(connection_error)

        transport = self._client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError(connection_error)

        stdout, stderr = bytearray(), bytearray()
        monotonic = time.monotonic
        deadline = None if timeout_s is None else monotonic() + timeout_s
        channel = transport.open_session(timeout=timeout_s)

        def drain() -> bool:
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
            return RemoteResult(
                exit_status=exit_status,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )

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
        client = self._client
        self._client = None
        if client is not None:
            client.close()


class SessionRegistry:
    """Keep live SSH clients behind string ids for graph/tool lookup."""

    def __init__(self) -> None:
        self._sessions: dict[str, RemoteSession] = {}

    def add(self, session: RemoteSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"remote session already exists: {session.session_id}")
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> RemoteSession:
        return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self.close(session_id)
