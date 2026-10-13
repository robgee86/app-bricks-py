# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""What a forked child does before and while it runs the app, as close as possible to `python main.py`. Stdlib only."""

import builtins
import glob
import importlib.machinery
import io
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from . import prepare


class Runner(StrEnum):
    PYTHON = "python"
    STREAMLIT = "streamlit"


@dataclass(frozen=True)
class Launch:
    """Everything the child needs to become the app."""

    app: str
    env: Mapping[str, str]
    """The supervisor environment with the requested one on top."""
    ports: Mapping[int, int]
    """The internal port the app binds for each public port the launcher forwards."""
    output_fd: int
    runner: Runner = Runner.PYTHON
    interpreter: str | None = None
    """The app venv interpreter, set only when the app has dependencies of its own."""
    streamlit_port: int | None = None
    """The public port Streamlit serves, shown in the URLs it prints; None leaves Streamlit its default."""

    @property
    def main_py(self) -> str:
        return os.path.join(self.app, "python", "main.py")

    @property
    def exec_argv(self) -> list[str] | None:
        """The command the child execs, None when it runs main.py warm in this process."""
        if self.runner is Runner.STREAMLIT:
            return [self.interpreter or sys.executable, "-m", "streamlit", "run", *self._streamlit_port_options, self.main_py]
        if self.interpreter:
            return [self.interpreter, self.main_py]
        return None

    @property
    def _streamlit_port_options(self) -> list[str]:
        # Streamlit binds the internal port of its public one, or the public one itself when it is not forwarded
        if self.streamlit_port is None:
            return []
        internal = self.ports.get(self.streamlit_port)
        if internal is None:
            return ["--server.port", str(self.streamlit_port)]
        return ["--server.port", str(internal), "--browser.serverPort", str(self.streamlit_port)]


def port_variable(public: int) -> str:
    """Names the variable telling the app the internal port to bind for a public one."""
    return f"ARDUINO_APP_PORT_{public}"


def runner(app: str) -> Runner:
    """Returns how the app runs: through Streamlit when its app.yaml uses the streamlit_ui brick, as run.sh decides."""
    return Runner.STREAMLIT if prepare.uses_streamlit(app) else Runner.PYTHON


def venv_interpreter(app: str) -> str | None:
    """Returns the interpreter of the app venv when the app has dependencies of its own, None when the image is enough."""
    python = prepare.venv_python(app)
    return python if os.path.exists(python) and (prepare.dependencies(app) or _venv_has_packages(app)) else None


def _venv_has_packages(app: str) -> bool:
    # An empty `uv venv` holds only _virtualenv.pth and _virtualenv.py
    for path in glob.glob(os.path.join(glob.escape(app), ".cache", ".venv", "lib", "python*", "site-packages", "*")):
        name = os.path.basename(path)
        if name.endswith(".dist-info") or (name.endswith(".pth") and name != "_virtualenv.pth"):
            return True
    return False


def environment(launch: Launch) -> dict[str, str]:
    """Builds the app environment: the launch one, an ARDUINO_APP_PORT_<public> per forwarded port and the run.sh additions."""
    env = {**launch.env, **{port_variable(public): str(internal) for public, internal in launch.ports.items()}, "PYTHONUNBUFFERED": "1"}
    bricks = os.path.join(launch.app, "bricks")
    if os.path.isdir(bricks):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), bricks]))
    if launch.interpreter:
        venv = os.path.dirname(os.path.dirname(launch.interpreter))
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = os.pathsep.join(filter(None, [os.path.join(venv, "bin"), env.get("PATH")]))
    return env


def detach(output_fd: int) -> None:
    """Makes the freshly forked process a group leader writing only to its output pipe, first thing after the fork."""
    os.setsid()
    # Inherited fds are pointed at /dev/null, not closed: the supervisor objects still own their numbers
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in _open_fds():
        if fd > 2 and fd not in (output_fd, devnull):
            os.dup2(devnull, fd, inheritable=False)
    os.dup2(output_fd, 1)
    os.dup2(output_fd, 2)
    os.close(output_fd)
    os.close(devnull)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)


def run(launch: Launch) -> int:
    """Turns the detached child into the app and runs it; returns its exit code.

    Raises:
        SystemExit: when the app exits through sys.exit(), as it would in its own interpreter.
    """
    env = environment(launch)
    os.environ.clear()
    os.environ.update(env)
    os.chdir(launch.app)
    argv = launch.exec_argv
    if argv is not None:
        os.execve(argv[0], argv, env)
    sys.argv = [launch.main_py]
    # As a fresh interpreter: the script folder first, then PYTHONPATH
    sys.path[0] = os.path.dirname(launch.main_py)
    sys.path[1:1] = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path and path not in sys.path]
    return _exec_main(launch.main_py)


def _open_fds() -> list[int]:
    listing = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    fds: list[int] = []
    for name in os.listdir(listing):
        try:
            os.fstat(int(name))
        except OSError:
            continue
        fds.append(int(name))
    return fds


def _exec_main(main_py: str) -> int:
    # A permanent module, not runpy: the app's threads and atexit handlers must keep seeing it as __main__
    module = types.ModuleType("__main__")
    module.__file__ = main_py
    module.__loader__ = importlib.machinery.SourceFileLoader("__main__", main_py)
    module.__spec__ = None
    module.__package__ = None
    module.__dict__["__cached__"] = None
    module.__dict__["__builtins__"] = builtins
    sys.modules["__main__"] = module
    try:
        with open(main_py, "rb") as f:
            code = compile(f.read(), main_py, "exec")
        exec(code, module.__dict__)  # noqa: S102
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
            tb = tb.tb_next
        # The default hook prints exc.__traceback__ whatever traceback it gets
        sys.excepthook(type(exc), exc.with_traceback(tb), tb)
        return 1
    finally:
        # As the interpreter does for a script: atexit handlers no longer see them
        module.__dict__.pop("__file__", None)
        module.__dict__.pop("__cached__", None)
    return 0
