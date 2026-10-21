# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The launcher process: imports the heavy libraries once, then forks the app of its container on request.

The parent stays single-threaded: one asyncio loop, no executors, numeric addresses only. Forks happen
between two runs of the loop, never inside a callback, so the child never unwinds through asyncio.
"""

import asyncio
import gc
import importlib
import importlib.util
import logging
import os
import signal
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import FrameType
from typing import Any, NoReturn

from arduino.version import __version__

from . import child, prepare, protocol
from .child import Launch, Runner
from .protocol import ErrorCode, LauncherError, Ping, Request, Start, Status, Stop

logger = logging.getLogger("arduino.app_tools.app_launcher")

# The peripherals import their backends eagerly: each package brings its dependencies into the warm interpreter
DEFAULT_PRELOAD = ",".join((
    "arduino.app_utils",
    "arduino.app_peripherals.camera",
    "arduino.app_peripherals.microphone",
    "arduino.app_peripherals.speaker",
    "arduino.app_bricks.web_ui",
))
DEFAULT_RUN_SH = "/run.sh"
REQUEST_TIMEOUT_S = 5.0
HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD)
STRAY_SWEEPS = 5
PRELOAD_GRACE_S = 0.5
"""How long the preload waits once the socket is open: the start a fresh container is brought up for comes first."""
"""How many times a sweep lists and kills what a stopped app left behind, for a leftover forking in between."""
_HOST_PID_NAMESPACE = 0xEFFFFFFC
"""The inode the kernel always gives the initial pid namespace, the host one."""
VENV_PRELOAD = ("streamlit.web.bootstrap",)
"""What a launcher running in an app venv preloads on top, when the venv has it: the Streamlit of the app."""
LISTEN_FD_VAR, PENDING_FD_VAR, PENDING_REQUEST_VAR = "APP_LAUNCHER_LISTEN_FD", "APP_LAUNCHER_PENDING_FD", "APP_LAUNCHER_PENDING_REQUEST"
PENDING_PREPARED_VAR = "APP_LAUNCHER_PENDING_PREPARED"

SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


@dataclass(frozen=True)
class Config:
    """Launcher settings, from the APP_LAUNCHER_* environment variables."""

    preload: tuple[str, ...]
    socket_path: str
    stop_timeout_s: float
    run_sh: str
    prepare_timeout_s: float

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        """Reads the settings.

        Raises:
            ValueError: when a value is malformed.
        """
        return cls(
            preload=tuple(name.strip() for name in env.get("APP_LAUNCHER_PRELOAD", DEFAULT_PRELOAD).split(",") if name.strip()),
            socket_path=env.get("APP_LAUNCHER_SOCKET", protocol.DEFAULT_SOCKET),
            stop_timeout_s=float(env.get("APP_LAUNCHER_STOP_TIMEOUT_S", "5")),
            run_sh=env.get("APP_LAUNCHER_RUN_SH", DEFAULT_RUN_SH),
            prepare_timeout_s=float(env.get("APP_LAUNCHER_PREPARE_TIMEOUT_S", "600")),
        )


@dataclass(frozen=True)
class Handover:
    """What a launcher that became the venv interpreter of its app took along: its socket and the start it was serving."""

    listen_fd: int | None = None
    pending_fd: int | None = None
    pending_request: bytes | None = None
    pending_prepared: bool = False
    """Whether the launcher prepared the app of the pending start before it became the venv interpreter."""

    @classmethod
    def take_from(cls, env: MutableMapping[str, str]) -> "Handover":
        """Reads and removes the handover from the environment, so that no app inherits it."""
        listen_fd, pending_fd, pending, prepared = (
            env.pop(name, None) for name in (LISTEN_FD_VAR, PENDING_FD_VAR, PENDING_REQUEST_VAR, PENDING_PREPARED_VAR)
        )
        return cls(
            listen_fd=None if listen_fd is None else int(listen_fd),
            pending_fd=None if pending_fd is None else int(pending_fd),
            pending_request=None if pending is None else pending.encode(),
            pending_prepared=prepared == "1",
        )

    @property
    def happened(self) -> bool:
        return self.listen_fd is not None


@dataclass(eq=False)
class Child:
    """The running app process."""

    app: str
    pid: int
    runner: Runner
    warm: bool
    started_at: datetime
    exited: asyncio.Future[int]
    """Resolved with the wait status once the process is reaped."""
    stopping: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "pid": self.pid,
            "runner": str(self.runner),
            "warm": self.warm,
            "started_at": self.started_at.isoformat(),
            "stopping": self.stopping,
        }


@dataclass(frozen=True)
class _Preparation:
    """A running `run.sh prepare`."""

    pid: int
    exited: asyncio.Future[int]


@dataclass(frozen=True)
class _ForkRequest:
    launch: Launch
    done: asyncio.Future[Child] = field(repr=False)


class _BecomeVenv(Exception):
    """A start of an app with a venv of its own: the launcher has to become that interpreter before it forks."""

    def __init__(self, interpreter: str, request: Start, prepared: bool) -> None:
        super().__init__(interpreter)
        self.interpreter = interpreter
        self.request = request
        self.prepared = prepared


class Supervisor:
    """Owns the control socket and the app process of this container."""

    def __init__(self, config: Config, handover: Handover | None = None) -> None:
        """Prepares the supervisor; the signal dispositions of this moment are the ones a fresh `python main.py` has."""
        self._config = config
        self._handover = handover or Handover()
        self._dispositions: dict[int, SignalHandler] = {signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS}
        self._pending_preload = list(config.preload)
        self._preloaded: list[str] = []
        self._preload_failed: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._lifecycle = asyncio.Lock()
        self._child: Child | None = None
        self._preparation: _Preparation | None = None
        self._control: asyncio.Server | None = None
        self._fork_request: _ForkRequest | None = None
        self._owns_container = _owns_container()
        self._exiting = False
        self._exit_task: asyncio.Task[None] | None = None
        self._started = time.monotonic()

    def serve(self) -> Launch | None:
        """Runs until SIGTERM or SIGINT.

        Returns:
            Launch | None: None in the supervisor once it is done; in a forked child, the app it has to become.
        """
        self._loop.add_signal_handler(signal.SIGTERM, self._request_exit, "SIGTERM")
        self._loop.add_signal_handler(signal.SIGINT, self._request_exit, "SIGINT")
        self._loop.add_signal_handler(signal.SIGCHLD, self._reap)
        self._loop.run_until_complete(self._open())
        self._log_ready()
        if self._handover.pending_fd is not None and self._handover.pending_request is not None:
            self._loop.create_task(self._serve_pending(self._handover.pending_fd, self._handover.pending_request))
        # The socket is served first: a start arriving before the preload is done forks between two imports, the app
        # imports what it needs itself, and the remaining modules are imported afterwards, one between two turns of the loop
        self._loop.call_later(PRELOAD_GRACE_S, self._preload_step)
        while True:
            self._loop.run_forever()
            request, self._fork_request = self._fork_request, None
            if request is None:
                break
            if self._fork(request) == 0:
                return request.launch
        self._loop.run_until_complete(self._close())
        self._loop.close()
        logger.info("Exited")
        return None

    async def _open(self) -> None:
        if self._handover.listen_fd is not None:
            sock = socket.socket(fileno=self._handover.listen_fd)
        else:
            sock = self._bind_control_socket()
        self._control = await asyncio.start_unix_server(self._on_control, sock=sock, limit=protocol.MAX_LINE_BYTES)

    def _bind_control_socket(self) -> socket.socket:
        path = self._config.socket_path
        with suppress(FileNotFoundError):
            if stat.S_ISSOCK(os.stat(path).st_mode):
                os.unlink(path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        umask = os.umask(0o117)
        try:
            sock.bind(path)
        except OSError:
            sock.close()
            raise
        finally:
            os.umask(umask)
        return sock

    async def _close(self) -> None:
        if self._control is not None:
            self._control.close()
        with suppress(FileNotFoundError):
            os.unlink(self._config.socket_path)

    def _log_ready(self) -> None:
        threads = [thread.name for thread in threading.enumerate()]
        if len(threads) > 1:
            logger.warning(f"Threads alive after the preload, starting apps will be refused: {threads}")
        # Native threads are invisible to threading: only Linux lists them all
        native = f", {len(os.listdir('/proc/self/task'))} OS threads" if os.path.isdir("/proc/self/task") else ""
        sweep = "" if self._owns_container else ", not atop a container: a stop kills the app process group only"
        venv = f", in the venv {sys.prefix}" if _in_a_venv() else ""
        logger.info(f"Ready on {self._config.socket_path}, {len(threads)} Python threads{native}{sweep}{venv}")
        if self._in_app_venv():
            # The fork inherits the app's own packages too: Streamlit, when the venv has it
            self._pending_preload.extend(
                name for name in VENV_PRELOAD if name not in self._pending_preload and importlib.util.find_spec(name.partition(".")[0])
            )

    def _in_app_venv(self) -> bool:
        """Whether this interpreter is the venv of the app of this container: named by APP_DIR, or by the start taken along."""
        apps = [os.environ.get("APP_DIR")]
        if self._handover.pending_request is not None:
            with suppress(LauncherError):
                request = protocol.decode_request(self._handover.pending_request)
                apps.append(request.app if isinstance(request, Start) else None)
        return any(app and child.in_venv(prepare.venv_dir(app)) for app in apps)

    def _preload_step(self) -> None:
        """Imports the next module of the preload, keeping any signal handler it installs for the app, not for the loop."""
        if self._exiting or not self._pending_preload:
            if not self._pending_preload and (threads := [thread.name for thread in threading.enumerate()])[1:]:
                logger.warning(f"Threads alive after the preload, starting apps will be refused: {threads}")
            return
        name = self._pending_preload.pop(0)
        loop_handlers = {signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS}
        # Nothing of the loop runs during an import: a stop arriving then exits at once, even as PID 1
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, _exit_now)
        started = time.monotonic()
        try:
            importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Preload of {name} failed: {e!r}")
            self._preload_failed.append(name)
        else:
            self._preloaded.append(name)
            logger.info(f"Preloaded {name} in {_ms_since(started)} ms")
        for signum, handler in loop_handlers.items():
            if (installed := signal.getsignal(signum)) not in (handler, _exit_now):
                self._dispositions[signum] = installed
            signal.signal(signum, handler)
        self._loop.call_soon(self._preload_step)

    def _request_exit(self, signame: str) -> None:
        if self._exiting:
            return
        logger.info(f"{signame} received, stopping")
        self._exiting = True
        self._stop_preparation()
        self._exit_task = self._loop.create_task(self._exit())

    async def _exit(self) -> None:
        async with self._lifecycle:
            await self._stop_child()
        self._loop.stop()

    # Control socket

    async def _on_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await _read_request(reader)
        except LauncherError as e:
            await self._write(writer, protocol.error(e))
            return
        except (OSError, TimeoutError):
            writer.close()
            return
        await self._respond(writer, request)

    async def _serve_pending(self, fd: int, line: bytes) -> None:
        """Answers the start the launcher was serving when it became the venv interpreter, on the connection it took along."""
        _, writer = await asyncio.open_unix_connection(sock=socket.socket(fileno=fd))
        await self._respond(writer, protocol.decode_request(line), prepared_before=self._handover.pending_prepared)

    async def _respond(self, writer: asyncio.StreamWriter, request: Request, prepared_before: bool = False) -> None:
        try:
            response = await self._handle(request, prepared_before)
        except _BecomeVenv as e:
            self._become_venv(e, writer)
        except LauncherError as e:
            response = protocol.error(e)
        except Exception as e:
            logger.exception("Request failed")
            response = protocol.error(LauncherError(ErrorCode.INTERNAL, repr(e)))
        await self._write(writer, response)

    async def _write(self, writer: asyncio.StreamWriter, response: dict[str, Any]) -> None:
        try:
            writer.write(protocol.encode(response))
            await writer.drain()
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()

    def _become_venv(self, become: _BecomeVenv, writer: asyncio.StreamWriter) -> NoReturn:
        """Replaces this process with the launcher under the venv interpreter of the app, the socket and the pending start
        along: the apps forked from now on inherit the packages of the venv. The pid stays, nothing of the loop survives."""
        assert self._control is not None
        listen_fd = os.dup(self._control.sockets[0].fileno())
        pending_fd = os.dup(writer.get_extra_info("socket").fileno())
        for fd in (listen_fd, pending_fd):
            os.set_inheritable(fd, True)
        interpreter, request = become.interpreter, become.request
        env = {
            **os.environ,
            LISTEN_FD_VAR: str(listen_fd),
            PENDING_FD_VAR: str(pending_fd),
            PENDING_REQUEST_VAR: protocol.encode({"cmd": "start", "app": request.app, "env": dict(request.env)}).decode(),
            PENDING_PREPARED_VAR: "1" if become.prepared else "0",
        }
        logger.info(f"Becoming the venv interpreter {interpreter} of {request.app}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(interpreter, [interpreter, "-c", _VENV_BOOTSTRAP % _package_root()], env)

    async def _handle(self, request: Request, prepared_before: bool = False) -> dict[str, Any]:
        match request:
            case Ping():
                return protocol.ok()
            case Status():
                return protocol.ok(**self._status())
            case Start():
                async with self._lifecycle:
                    self._refuse_while_exiting()
                    return await self._start(request, prepared_before)
            case Stop():
                # A preparation may hold the lifecycle for minutes
                self._stop_preparation()
                async with self._lifecycle:
                    self._refuse_while_exiting()
                    stopped = await self._stop_child()
                    return protocol.ok(stopped=[] if stopped is None else [stopped])

    def _refuse_while_exiting(self) -> None:
        if self._exiting:
            raise LauncherError(ErrorCode.BUSY, "the launcher is shutting down")

    def _status(self) -> dict[str, Any]:
        return {
            "child": None if self._child is None else self._child.describe(),
            "venv": sys.prefix if _in_a_venv() else None,
            "preload": {"modules": list(self._config.preload), "loaded": list(self._preloaded), "failed": list(self._preload_failed)},
            "version": __version__,
            "uptime_s": round(time.monotonic() - self._started, 3),
        }

    # Lifecycle

    async def _start(self, request: Start, prepared_before: bool = False) -> dict[str, Any]:
        """Starts the app; prepared_before tells that this very start prepared it already, in the process this one replaced."""
        app = request.app
        if not os.path.isdir(app):
            raise LauncherError(ErrorCode.NOT_FOUND, f"no app folder at {app}")
        if not os.path.isfile(os.path.join(app, "python", "main.py")):
            raise LauncherError(ErrorCode.NOT_FOUND, f"no python/main.py in {app}")

        started = time.monotonic()
        await self._stop_child()
        stop_ms = _ms_since(started)

        env = {**os.environ, **request.env}
        started = time.monotonic()
        prepared = await self._prepare(app, env) or prepared_before
        prepare_ms = _ms_since(started)

        interpreter = child.venv_interpreter(app)
        if interpreter is not None and not self._handover.happened:
            raise _BecomeVenv(interpreter, request, prepared)
        if interpreter is not None:
            logger.warning(f"Not running in the venv {interpreter} after becoming it, {app} runs cold in it")

        started = time.monotonic()
        launch = Launch(app=app, env=env, runner=child.runner(app), interpreter=interpreter)
        fork = _ForkRequest(launch=launch, done=self._loop.create_future())
        self._fork_request = fork
        self._loop.stop()
        started_child = await fork.done
        return protocol.ok(
            pid=started_child.pid,
            started_at=started_child.started_at.isoformat(),
            warm=started_child.warm,
            runner=str(started_child.runner),
            prepared=prepared,
            timings={"stop_ms": stop_ms, "prepare_ms": prepare_ms, "fork_ms": _ms_since(started)},
        )

    async def _prepare(self, app: str, env: Mapping[str, str]) -> bool:
        """Runs `run.sh prepare` when the app dependencies changed since the last success; returns whether it ran."""
        current = prepare.fingerprint(app)
        if current is None or not prepare.needs_preparation(app, current):
            return False
        try:
            # Checked in the loop step that spawns run.sh, so a shutdown never misses a preparation
            self._refuse_while_exiting()
            logger.info(f"Preparing {app}")
            await prepare.run(app, env, self._config.run_sh, self._config.prepare_timeout_s, self._watch_preparation)
        except LauncherError as e:
            if e.code is ErrorCode.PREPARE_FAILED:
                logger.warning(f"Preparation of {app} failed: {e.message.splitlines()[0]}")
            raise
        if prepare.recorded(app) != current:
            logger.warning(f"The preparation of {app} recorded nothing, the next start prepares it again")
        logger.info(f"Prepared {app}")
        return True

    def _watch_preparation(self, pid: int) -> asyncio.Future[int]:
        # Registered before the loop runs again, so the reaper always finds it
        self._preparation = _Preparation(pid=pid, exited=self._loop.create_future())
        return self._preparation.exited

    def _stop_preparation(self) -> None:
        if self._preparation is not None:
            logger.info("Stopping the preparation")
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(self._preparation.pid, signal.SIGTERM)

    def _fork(self, request: _ForkRequest) -> int:
        """Forks the app process; returns 0 in the child and the pid in the parent, or -1 when the fork failed."""
        if threading.active_count() > 1:
            threads = [thread.name for thread in threading.enumerate()]
            request.done.set_exception(LauncherError(ErrorCode.INTERNAL, f"refusing to fork with threads alive: {threads}"))
            return -1
        sys.stdout.flush()
        sys.stderr.flush()
        # Signals wait until the child has its own dispositions back
        signal.pthread_sigmask(signal.SIG_BLOCK, HANDLED_SIGNALS)
        # The child never collects what exists now, so none of the supervisor finalizers runs in the app
        gc.freeze()
        try:
            pid = os.fork()
        except OSError as e:
            pid = -1
            request.done.set_exception(LauncherError(ErrorCode.INTERNAL, f"fork failed: {e}"))
        if pid == 0:
            self._become_child()
        else:
            gc.unfreeze()
            if pid > 0:
                request.done.set_result(self._adopt(request.launch, pid))
        signal.pthread_sigmask(signal.SIG_UNBLOCK, HANDLED_SIGNALS)
        return pid

    def _adopt(self, launch: Launch, pid: int) -> Child:
        # Registered before the loop runs again, so the reaper always finds it
        self._child = Child(
            app=launch.app,
            pid=pid,
            runner=launch.runner,
            warm=launch.exec_argv is None,
            started_at=datetime.now(UTC),
            exited=self._loop.create_future(),
        )
        how = "warm" if self._child.warm else f"as {' '.join(launch.exec_argv or [])}"
        logger.info(f"Started {launch.app} as pid {pid}, {how}")
        return self._child

    def _become_child(self) -> None:
        # First, so the inherited fds, the event loop selector included, are the child's own before anything runs
        child.detach()
        self._loop.set_exception_handler(lambda loop, context: None)
        for signum, handler in self._dispositions.items():
            self._loop.remove_signal_handler(signum)
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)

    async def _stop_child(self) -> dict[str, Any] | None:
        """Stops the app, if one runs; returns how it ended."""
        if (target := self._child) is None:
            return None
        target.stopping = True
        _signal_group(target.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(target.exited), self._config.stop_timeout_s)
        except TimeoutError:
            logger.warning(f"{target.app} (pid {target.pid}) did not stop in {self._config.stop_timeout_s} s, killing it")
            _signal_group(target.pid, signal.SIGKILL)
            await target.exited
        return {"app": target.app, "pid": target.pid, **_exit_fields(target.exited.result())}

    def _reap(self) -> None:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            if (preparation := self._preparation) is not None and preparation.pid == pid:
                self._preparation = None
                self._kill_strays()
                preparation.exited.set_result(status)
            elif (reaped := self._child) is not None and reaped.pid == pid:
                self._child = None
                # What the app left behind dies with it, as in a container whose main process ends
                _signal_group(pid, signal.SIGKILL)
                self._kill_strays()
                fields = _exit_fields(status)
                how = f"was killed by signal {fields['signal']}" if fields["signal"] is not None else f"exited with code {fields['exit_code']}"
                logger.info(f"{reaped.app} (pid {pid}) {how}")
                reaped.exited.set_result(status)

    def _kill_strays(self) -> None:
        """Kills every process of the container but PID 1 and the supervisor, helpers in sessions of their own included.

        Only when the supervisor tops a pid namespace of its own: elsewhere the other processes are not the app's.
        """
        if not self._owns_container:
            return
        keep = {1, os.getpid(), *(p.pid for p in (self._child, self._preparation) if p is not None)}
        killed: set[int] = set()
        for _ in range(STRAY_SWEEPS):
            strays = [pid for pid in _live_pids() if pid not in keep]
            if not strays:
                break
            for pid in strays:
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
            killed.update(strays)
        if killed:
            logger.info(f"Killed what was left behind: pids {sorted(killed)}")


_VENV_BOOTSTRAP = """\
import sys
try:
    import arduino.app_tools.app_launcher
except ImportError:
    sys.path.append(%r)
from arduino.app_tools.app_launcher.__main__ import main
raise SystemExit(main())
"""
"""Runs the launcher in a venv that sees the image packages; a venv that does not, as in a test, gets this installation last."""


def _package_root() -> str:
    """The folder the `arduino` package is installed in."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _in_a_venv() -> bool:
    return sys.prefix != sys.base_prefix


def _owns_container() -> bool:
    """Whether the supervisor tops a pid namespace of its own, as PID 1 or the child of the init docker runs as PID 1."""
    try:
        nested = os.stat("/proc/self/ns/pid").st_ino != _HOST_PID_NAMESPACE
    except OSError:
        return False
    return nested and 1 in (os.getpid(), os.getppid())


def _signal_group(pid: int, signum: int) -> None:
    """Signals the process group the child leads, or the child alone when it has not become its leader yet."""
    for kill in (os.killpg, os.kill):
        try:
            kill(pid, signum)
            return
        except ProcessLookupError:
            continue
        except PermissionError:
            return


def _live_pids() -> list[int]:
    """The processes started in this pid namespace, from /proc: not the exited ones that only wait for their reaping, nor
    the ones a parent outside the namespace started, the `docker exec` sessions."""
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []
    live: list[int] = []
    for pid in pids:
        state, parent = _state(pid)
        if state not in ("", "Z", "X") and parent != 0:
            live.append(pid)
    return live


def _state(pid: int) -> tuple[str, int]:
    """The state letter and the parent pid of the process; the state is empty when it is gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            # The command name may hold spaces and parentheses: the state and the parent follow its last parenthesis
            state, parent = stat_file.read().rpartition(")")[2].split()[:2]
            return state, int(parent)
    except (OSError, ValueError):
        return "", 0


async def _read_request(reader: asyncio.StreamReader) -> Request:
    try:
        line = await asyncio.wait_for(reader.readline(), REQUEST_TIMEOUT_S)
    except ValueError:
        raise LauncherError(ErrorCode.BAD_REQUEST, f"a request must fit in {protocol.MAX_LINE_BYTES} bytes") from None
    return protocol.decode_request(line)


def _exit_fields(status: int) -> dict[str, int | None]:
    if os.WIFSIGNALED(status):
        return {"exit_code": None, "signal": os.WTERMSIG(status)}
    return {"exit_code": os.waitstatus_to_exitcode(status), "signal": None}


def _ms_since(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)


def _setup_logging() -> None:
    """Logs to stderr, which the container log shows next to the app output: warnings only unless APP_LAUNCHER_LOG_LEVEL says more."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s app-launcher: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("APP_LAUNCHER_LOG_LEVEL", "WARNING").upper())
    logger.propagate = False


def _exit_now(signum: int, frame: FrameType | None) -> NoReturn:
    raise SystemExit(0)


def main() -> int:
    """Runs the supervisor; in a forked child it returns the app's exit code."""
    _setup_logging()
    handover = Handover.take_from(os.environ)
    try:
        config = Config.from_env(os.environ)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 2
    launch = Supervisor(config, handover).serve()
    return 0 if launch is None else child.run(launch)
