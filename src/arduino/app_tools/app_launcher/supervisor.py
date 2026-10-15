# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The launcher process: imports the heavy libraries once, then forks a child per app on request.

The parent stays single-threaded: one asyncio loop, no executors, numeric addresses only. Forks happen
between two runs of the loop, never inside a callback, so the child never unwinds through asyncio.
"""

import asyncio
import gc
import importlib
import logging
import os
import signal
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import FrameType
from typing import Any, NoReturn

from arduino.version import __version__

from . import child, prepare, protocol
from .applog import CHUNK_BYTES, AppLog
from .child import Launch, Runner
from .protocol import ErrorCode, LauncherError, Mode, Ping, Request, Start, Status, Stop

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


@dataclass(eq=False)
class Child:
    """A running app process."""

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
            "state": "stopping" if self.stopping else "running",
            "started_at": self.started_at.isoformat(),
            "runner": str(self.runner),
            "warm": self.warm,
        }


@dataclass(frozen=True)
class _Preparation:
    """A running `run.sh prepare`."""

    app: str
    exited: asyncio.Future[int]


@dataclass(frozen=True)
class _ForkRequest:
    launch: Launch
    output_fd: int
    log: AppLog | None
    done: asyncio.Future[Child] = field(repr=False)


def preload(modules: Sequence[str]) -> list[str]:
    """Imports the modules one by one and returns those that failed."""
    failed: list[str] = []
    for name in modules:
        started = time.monotonic()
        try:
            importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Preload of {name} failed: {e!r}")
            failed.append(name)
            continue
        logger.info(f"Preloaded {name} in {_ms_since(started)} ms")
    return failed


class Supervisor:
    """Owns the control socket and the table of running apps."""

    def __init__(self, config: Config, preload_failed: Sequence[str], dispositions: Mapping[int, SignalHandler]) -> None:
        """Prepares the supervisor.

        Args:
            config (Config): The settings.
            preload_failed (Sequence[str]): The preloaded modules that failed to import, reported by status.
            dispositions (Mapping[int, SignalHandler]): The handlers of HANDLED_SIGNALS a fresh `python main.py` has.
        """
        self._config = config
        self._dispositions = dict(dispositions)
        self._preload_failed = tuple(preload_failed)
        self._loop = asyncio.new_event_loop()
        self._lifecycle = asyncio.Lock()
        self._children: dict[str, Child] = {}
        self._preparations: dict[int, _Preparation] = {}
        self._control: asyncio.Server | None = None
        self._fork_request: _ForkRequest | None = None
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
        self._control = await asyncio.start_unix_server(self._on_control, sock=self._bind_control_socket(), limit=protocol.MAX_LINE_BYTES)

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
        logger.info(f"Ready on {self._config.socket_path}, {len(threads)} Python threads{native}")

    def _request_exit(self, signame: str) -> None:
        if self._exiting:
            return
        logger.info(f"{signame} received, stopping")
        self._exiting = True
        self._stop_preparations(None)
        self._exit_task = self._loop.create_task(self._exit())

    async def _exit(self) -> None:
        async with self._lifecycle:
            await self._stop_children(list(self._children.values()))
        self._loop.stop()

    # Control socket

    async def _on_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                response = await self._handle(await _read_request(reader))
            except LauncherError as e:
                response = protocol.error(e)
            except Exception as e:
                logger.exception("Request failed")
                response = protocol.error(LauncherError(ErrorCode.INTERNAL, repr(e)))
            writer.write(protocol.encode(response))
            await writer.drain()
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()

    async def _handle(self, request: Request) -> dict[str, Any]:
        match request:
            case Ping():
                return protocol.ok()
            case Status():
                return protocol.ok(**self._status())
            case Start():
                async with self._lifecycle:
                    self._refuse_while_exiting()
                    return await self._start(request)
            case Stop():
                # A preparation may hold the lifecycle for minutes
                self._stop_preparations(request.app)
                async with self._lifecycle:
                    self._refuse_while_exiting()
                    targets = [c for c in self._children.values() if request.app in (None, c.app)]
                    return protocol.ok(stopped=await self._stop_children(targets))

    def _refuse_while_exiting(self) -> None:
        if self._exiting:
            raise LauncherError(ErrorCode.BUSY, "the launcher is shutting down")

    def _status(self) -> dict[str, Any]:
        return {
            "children": [c.describe() for c in self._children.values()],
            "preload": {"modules": list(self._config.preload), "failed": list(self._preload_failed)},
            "version": __version__,
            "uptime_s": round(time.monotonic() - self._started, 3),
        }

    # Lifecycle

    async def _start(self, request: Start) -> dict[str, Any]:
        if request.mode is Mode.ADD:
            raise LauncherError(ErrorCode.UNSUPPORTED, "only one app at a time is supported for now, use mode replace")
        app = request.app
        if not os.path.isdir(app):
            raise LauncherError(ErrorCode.NOT_FOUND, f"no app folder at {app}")
        if not os.path.isfile(os.path.join(app, "python", "main.py")):
            raise LauncherError(ErrorCode.NOT_FOUND, f"no python/main.py in {app}")

        started = time.monotonic()
        await self._stop_children(list(self._children.values()))
        stop_ms = _ms_since(started)

        env = {**os.environ, **request.env}
        log = self._open_log(app)
        started = time.monotonic()
        prepared = await self._prepare(app, env, log)
        prepare_ms = _ms_since(started)

        started = time.monotonic()
        output_fd, input_fd = os.pipe()
        os.set_blocking(output_fd, False)
        launch = Launch(app=app, env=env, output_fd=input_fd, runner=child.runner(app), interpreter=child.venv_interpreter(app))
        fork = _ForkRequest(launch=launch, output_fd=output_fd, log=log, done=self._loop.create_future())
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

    async def _prepare(self, app: str, env: Mapping[str, str], log: AppLog | None) -> bool:
        """Runs `run.sh prepare` when the app dependencies changed since the last success; returns whether it ran."""
        current = prepare.fingerprint(app)
        if current is None or not prepare.needs_preparation(app, current):
            return False
        try:
            # Checked in the loop step that spawns run.sh, so a shutdown never misses a preparation
            self._refuse_while_exiting()
            logger.info(f"Preparing {app}")
            await prepare.run(app, env, self._config.run_sh, self._config.prepare_timeout_s, log, lambda pid: self._watch_preparation(app, pid))
        except BaseException as e:
            if isinstance(e, LauncherError) and e.code is ErrorCode.PREPARE_FAILED:
                logger.warning(f"Preparation of {app} failed: {e.message.splitlines()[0]}")
            if log is not None:
                log.close()
            raise
        try:
            prepare.record(app, current)
        except OSError as e:
            logger.warning(f"Cannot record the preparation of {app}, the next start prepares it again: {e}")
        logger.info(f"Prepared {app}")
        return True

    def _watch_preparation(self, app: str, pid: int) -> asyncio.Future[int]:
        # Registered before the loop runs again, so the reaper always finds it
        preparation = _Preparation(app=app, exited=self._loop.create_future())
        self._preparations[pid] = preparation
        return preparation.exited

    def _stop_preparations(self, app: str | None) -> None:
        for pid, preparation in self._preparations.items():
            if app in (None, preparation.app):
                logger.info(f"Stopping the preparation of {preparation.app}")
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGTERM)

    def _fork(self, request: _ForkRequest) -> int:
        """Forks the app process; returns 0 in the child and the pid in the parent, or -1 when the fork failed."""
        if threading.active_count() > 1:
            threads = [thread.name for thread in threading.enumerate()]
            self._abandon(request, LauncherError(ErrorCode.INTERNAL, f"refusing to fork with threads alive: {threads}"))
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
            self._abandon(request, LauncherError(ErrorCode.INTERNAL, f"fork failed: {e}"))
        if pid == 0:
            self._become_child(request.launch.output_fd)
        else:
            gc.unfreeze()
            if pid > 0:
                request.done.set_result(self._adopt(request, pid))
        signal.pthread_sigmask(signal.SIG_UNBLOCK, HANDLED_SIGNALS)
        return pid

    def _adopt(self, request: _ForkRequest, pid: int) -> Child:
        # Registered before the loop runs again, so the reaper always finds it
        os.close(request.launch.output_fd)
        self._loop.add_reader(request.output_fd, self._relay_output, request.output_fd, request.log)
        launched = Child(
            app=request.launch.app,
            pid=pid,
            runner=request.launch.runner,
            warm=request.launch.exec_argv is None,
            started_at=datetime.now(UTC),
            exited=self._loop.create_future(),
        )
        self._children[launched.app] = launched
        how = "warm" if launched.warm else f"as {' '.join(request.launch.exec_argv or [])}"
        logger.info(f"Started {launched.app} as pid {pid}, {how}")
        return launched

    def _abandon(self, request: _ForkRequest, error: LauncherError) -> None:
        os.close(request.launch.output_fd)
        os.close(request.output_fd)
        if request.log is not None:
            request.log.close()
        request.done.set_exception(error)

    def _become_child(self, output_fd: int) -> None:
        # First, so the inherited fds, the event loop selector included, are the child's own before anything runs
        child.detach(output_fd)
        self._loop.set_exception_handler(lambda loop, context: None)
        for signum, handler in self._dispositions.items():
            self._loop.remove_signal_handler(signum)
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)

    async def _stop_children(self, targets: Sequence[Child]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._stop(target) for target in targets)))

    async def _stop(self, target: Child) -> dict[str, Any]:
        target.stopping = True
        with suppress(ProcessLookupError):
            os.kill(target.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(target.exited), self._config.stop_timeout_s)
        except TimeoutError:
            logger.warning(f"{target.app} (pid {target.pid}) did not stop in {self._config.stop_timeout_s} s, killing it")
            for kill in (os.killpg, os.kill):
                with suppress(ProcessLookupError, PermissionError):
                    kill(target.pid, signal.SIGKILL)
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
            if (preparation := self._preparations.pop(pid, None)) is not None:
                if not preparation.exited.done():
                    preparation.exited.set_result(status)
                continue
            reaped = next((c for c in self._children.values() if c.pid == pid), None)
            if reaped is None:
                continue
            del self._children[reaped.app]
            # What the app left behind dies with it, as in a container whose main process ends
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
            fields = _exit_fields(status)
            how = f"was killed by signal {fields['signal']}" if fields["signal"] is not None else f"exited with code {fields['exit_code']}"
            logger.info(f"{reaped.app} (pid {pid}) {how}")
            reaped.exited.set_result(status)

    # App output

    def _open_log(self, app: str) -> AppLog | None:
        # A start begins a new log: what the user reads after a refresh is this run only
        try:
            return AppLog(os.path.join(app, ".cache", "logs", "app.log"), fresh=True)
        except OSError as e:
            logger.warning(f"Cannot write the log of {app}, its output is dropped: {e}")
            return None

    def _relay_output(self, fd: int, log: AppLog | None) -> None:
        try:
            data = os.read(fd, CHUNK_BYTES)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if data:
            if log is not None:
                try:
                    log.write(data)
                except OSError as e:
                    logger.warning(f"Cannot write {log.path}: {e}")
            return
        self._loop.remove_reader(fd)
        os.close(fd)
        if log is not None:
            log.close()


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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s app-launcher: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _exit_during_preload(signum: int, frame: FrameType | None) -> NoReturn:
    raise SystemExit(0)


def _app_dispositions(before_preload: Mapping[int, SignalHandler]) -> dict[int, SignalHandler]:
    current = {signum: signal.getsignal(signum) for signum in HANDLED_SIGNALS}
    return {signum: before_preload.get(signum) if handler is _exit_during_preload else handler for signum, handler in current.items()}


def main() -> int:
    """Runs the supervisor; in a forked child it returns the app's exit code."""
    # Nothing to stop before the event loop takes over: a stop during the preload just exits, even as PID 1
    before_preload: dict[int, SignalHandler] = {signum: signal.signal(signum, _exit_during_preload) for signum in (signal.SIGTERM, signal.SIGINT)}
    _setup_logging()
    try:
        config = Config.from_env(os.environ)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 2
    failed = preload(config.preload)
    supervisor = Supervisor(config, failed, _app_dispositions(before_preload))
    launch = supervisor.serve()
    return 0 if launch is None else child.run(launch)
