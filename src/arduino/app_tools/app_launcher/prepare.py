# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Installs what an app depends on beyond the image through `run.sh prepare`, only when it changed. Stdlib only."""

import asyncio
import glob
import hashlib
import logging
import os
import sys
import signal
from collections.abc import Callable, Mapping
from contextlib import suppress

from .applog import CHUNK_BYTES, AppLog
from .protocol import ErrorCode, LauncherError

logger = logging.getLogger("arduino.app_tools.app_launcher")

STREAMLIT_UI = "arduino:streamlit_ui"
TAIL_BYTES = 8 * 1024
TAIL_LINES = 20

Watch = Callable[[int], asyncio.Future[int]]
"""Registers a spawned pid with the supervisor reaper, which resolves the future with its wait status."""


def uses_streamlit(app: str) -> bool:
    """Tells whether the app.yaml uses the streamlit_ui brick, as run.sh checks."""
    try:
        with open(os.path.join(app, "app.yaml"), encoding="utf-8") as f:
            return STREAMLIT_UI in f.read()
    except OSError:
        return False


def dependencies(app: str) -> list[tuple[str, bytes]]:
    """Returns what the app needs beyond the image as (name, content) pairs, empty when the image is enough."""
    found: list[tuple[str, bytes]] = []
    pattern = os.path.join(glob.escape(app), "bricks", "*", "requirements.txt")
    for path in [os.path.join(app, "python", "requirements.txt"), *sorted(glob.glob(pattern))]:
        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError:
            continue
        if content.strip():
            found.append((os.path.relpath(path, app), content))
    # run.sh renames each installed wheel to x.whl.installed
    with suppress(OSError):
        wheels = {name.removesuffix(".installed") for name in os.listdir(os.path.join(app, "python-libraries"))}
        found.extend((f"python-libraries/{name}", b"") for name in sorted(wheels) if name.endswith(".whl"))
    if uses_streamlit(app):
        found.append(("app.yaml", STREAMLIT_UI.encode()))
    return found


def fingerprint(app: str) -> str | None:
    """Returns the digest of the app dependencies, None when it has none of its own."""
    found = dependencies(app)
    if not found:
        return None
    digest = hashlib.sha256()
    for name, content in found:
        digest.update(f"{name}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


def venv_python(app: str) -> str:
    """Returns the interpreter path of the app venv, which run.sh creates."""
    return os.path.join(app, ".cache", ".venv", "bin", "python")


def needs_preparation(app: str, current: str) -> bool:
    """Tells whether the app with the `current` dependencies fingerprint has to be prepared before it starts."""
    return not os.path.exists(venv_python(app)) or recorded(app) != current


def recorded(app: str) -> str | None:
    """Returns the fingerprint recorded by the last successful preparation, None when there is none."""
    try:
        with open(_record_path(app), encoding="ascii") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def record(app: str, current: str) -> None:
    """Records a successful preparation.

    Raises:
        OSError: when the record cannot be written.
    """
    path = _record_path(app)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="ascii") as f:
        f.write(f"{current}\n")
    os.replace(temporary, path)


async def run(app: str, env: Mapping[str, str], run_sh: str, timeout_s: float, log: AppLog | None, watch: Watch, confine: bool = True) -> None:
    """Runs `run.sh prepare` on the app in its own process group, its output appended to the app log.

    The preparation runs app code, the install steps of its dependencies: confined to the app folder like the app itself,
    through the confinement module as a command, since posix_spawn runs nothing of ours before the exec.

    Raises:
        LauncherError: prepare_failed when run.sh cannot start, fails, is stopped or exceeds the timeout.
    """
    argv = [run_sh, "prepare"]
    if confine:
        # Isolated: the wrapper imports from the image only, nothing from its working directory, the environment or a user site
        argv = [sys.executable, "-I", "-m", "arduino.app_tools.app_launcher.confinement", "--app", app, "--", *argv]
    output_fd, input_fd = os.pipe()
    try:
        # posix_spawn, not a subprocess helper: only the supervisor reaper waits for its processes
        pid = os.posix_spawn(
            argv[0],
            argv,
            {**env, "APP_DIR": app},
            file_actions=[
                (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
                (os.POSIX_SPAWN_DUP2, input_fd, 1),
                (os.POSIX_SPAWN_DUP2, input_fd, 2),
            ],
            setpgroup=0,
        )
    except OSError as e:
        os.close(output_fd)
        raise LauncherError(ErrorCode.PREPARE_FAILED, f"cannot run {run_sh}: {e}") from None
    finally:
        os.close(input_fd)
    exited = watch(pid)
    tail = bytearray()
    try:
        async with asyncio.timeout(timeout_s):
            await _relay(output_fd, log, tail)
            status = await asyncio.shield(exited)
    except TimeoutError:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        await exited
        raise LauncherError(ErrorCode.PREPARE_FAILED, f"{run_sh} prepare timed out after {timeout_s:g} s{_last_lines(tail)}") from None
    if os.WIFSIGNALED(status):
        raise LauncherError(ErrorCode.PREPARE_FAILED, f"{run_sh} prepare was killed by signal {os.WTERMSIG(status)}{_last_lines(tail)}")
    if code := os.waitstatus_to_exitcode(status):
        raise LauncherError(ErrorCode.PREPARE_FAILED, f"{run_sh} prepare exited with code {code}{_last_lines(tail)}")


def _record_path(app: str) -> str:
    return os.path.join(app, ".cache", "launcher", "prepared.sha")


async def _relay(fd: int, log: AppLog | None, tail: bytearray) -> None:
    # The log stays open: the app writes to it next
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(fd, "rb", buffering=0))
    try:
        while data := await reader.read(CHUNK_BYTES):
            tail += data
            del tail[:-TAIL_BYTES]
            if log is not None:
                try:
                    log.write(data)
                except OSError as e:
                    logger.warning(f"Cannot write {log.path}: {e}")
    finally:
        transport.close()


def _last_lines(tail: bytearray) -> str:
    lines = tail.decode(errors="replace").splitlines()[-TAIL_LINES:]
    return ":\n" + "\n".join(lines) if lines else ""
