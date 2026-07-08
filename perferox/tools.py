import os
import shlex
import signal
import subprocess

from langchain_core.tools import tool

from perferox.prompts import RUNPODCTL_PROMPT
from perferox.remote import RemoteSession

DEFAULT_TIMEOUT_S = 30.0
MAX_OUTPUT_CHARS = 10000

@tool(description=RUNPODCTL_PROMPT)
def runpodctl(command: str) -> str:
  """Run one local runpodctl command and return its exit code and output."""
  try:
    args = shlex.split(command)
  except ValueError as exc:
    return f"invalid runpodctl command: {exc}"
  if args[:1] == ["runpodctl"]:
    args = args[1:]
  try:
    result = subprocess.run(["runpodctl", *args], text=True, capture_output=True)
  except FileNotFoundError:
    return "runpodctl not found on PATH"
  return _format_result(result.returncode, result.stdout, result.stderr)


def local_terminal(cwd: str = ""):
  """Build one shell tool for local coding-agent work."""

  @tool("local_terminal")
  def terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
    """Run one local shell command in a fresh bash process."""
    return _run_local(command, cwd, timeout_s)

  return terminal


def remote_terminal(session: RemoteSession, cwd: str = ""):
  """Build one shell tool for SSH-backed coding-agent work."""

  @tool("remote_terminal")
  def terminal(command: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> str:
    """Run one remote shell command through the persistent SSH client."""
    return _run_remote(session, command, cwd, timeout_s)

  return terminal


def _run_local(command: str, cwd: str, timeout_s: float | None) -> str:
  """Run a local command, killing the process group on timeout."""
  try:
    process = subprocess.Popen(
      ["bash", "-lc", command],
      cwd=cwd or os.getcwd(),
      text=True,
      encoding="utf-8",
      errors="replace",
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      start_new_session=os.name == "posix",
    )
    stdout, stderr = process.communicate(timeout=timeout_s)
    return _format_result(process.returncode, stdout, stderr)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
    stdout, stderr = process.communicate()
    return _format_result(None, stdout, f"{stderr}\ntimed out after {timeout_s}s")
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")


def _run_remote(session: RemoteSession, command: str, cwd: str, timeout_s: float | None) -> str:
  """Run a command through SSH, using cwd by prefixing the remote shell command."""
  if cwd:
    command = f"cd {shlex.quote(cwd)} && {command}"
  try:
    result = session.run(f"bash -lc {shlex.quote(command)}", timeout_s=timeout_s)
  except Exception as exc:
    return _format_result(None, "", f"{type(exc).__name__}: {exc}")
  return _format_result(result.exit_status, result.stdout, result.stderr)


def _format_result(exit_status: int | None, stdout: str, stderr: str) -> str:
  """Format command output for a tool message."""
  output = "\n".join(part for part in (stdout, stderr) if part)
  if len(output) > MAX_OUTPUT_CHARS:
    keep = MAX_OUTPUT_CHARS // 2
    skipped = len(output) - MAX_OUTPUT_CHARS
    output = f"{output[:keep]}\n\n... {skipped} chars elided ...\n\n{output[-keep:]}"
  return f"exit_code={exit_status}\n{output}"
