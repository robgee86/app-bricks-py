# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from arduino.app_tools.app_launcher import client

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher forks, POSIX only")

FAKE_MAIN = """
import os, signal, sys
print("hello from", __name__, sys.argv[0], os.environ["GREETING"], os.environ["INHERITED"], flush=True)
print("warning from", sys.argv[0], file=sys.stderr, flush=True)
signal.pause()
"""

# An app that leaves a helper behind in a session of its own, out of the app's process group, tagged by MARKER
LINGERING_MAIN = """
import os, signal, subprocess, sys
subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()", os.environ["MARKER"]], start_new_session=True)
print("helper started", os.environ["MARKER"], flush=True)
signal.pause()
"""

# Stands in for Streamlit: prints what `python -m streamlit` got
FAKE_STREAMLIT = """
import os, signal, sys
print("streamlit", *sys.argv[1:], os.getcwd(), flush=True)
signal.pause()
"""

# Stands in for `run.sh prepare`: a venv on the test interpreter, a marker per run, a failure on request
FAKE_RUN_SH = """#!/bin/sh
echo "installing $APP_DIR/python/requirements.txt"
if grep -qs missing-package "$APP_DIR/python/requirements.txt"; then
  echo "No solution found for missing-package"
  exit 1
fi
mkdir -p "$APP_DIR/.cache/.venv/bin"
ln -sf "{python}" "$APP_DIR/.cache/.venv/bin/python"
echo prepared >> "$APP_DIR/prepared.marker"
echo "prepared with GREETING=$GREETING INHERITED=$INHERITED"
"""

# Runs the launcher atop a pid namespace of its own, as in its container; unshare forwards no signal, it only waits
UNSHARE = ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "--kill-child"]


def can_unshare():
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    return subprocess.run([*UNSHARE, "true"], capture_output=True).returncode == 0


def wait_until(condition, timeout=5.0, details=lambda: ""):
    deadline = time.monotonic() + timeout
    while not (result := condition()):
        assert time.monotonic() < deadline, f"timed out {details()}"
        time.sleep(0.02)
    return result


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def tagged(marker):
    """The pids whose last argument is the marker, as seen from the test: from /proc on Linux, from ps elsewhere."""
    if not os.path.isdir("/proc"):
        listing = subprocess.run(["ps", "-A", "-ww", "-o", "pid=", "-o", "args="], capture_output=True, text=True, check=True).stdout
        return [int(pid) for pid, _, args in (line.strip().partition(" ") for line in listing.splitlines()) if args.endswith(f" {marker}")]
    return [pid for pid in proc_pids() if read_proc(pid, "cmdline").split("\0")[-2:-1] == [marker]]


def proc_pids():
    return [int(name) for name in os.listdir("/proc") if name.isdigit()]


def read_proc(pid, name):
    try:
        with open(f"/proc/{pid}/{name}", encoding="utf-8", errors="replace") as file:
            return file.read()
    except OSError:
        return ""


class Launcher:
    def __init__(self, preload="json", wrapper=(), extra_modules=None):
        # AF_UNIX paths are limited to about 100 bytes
        self.root = Path(tempfile.mkdtemp(prefix="launcher", dir="/tmp")).resolve()
        self.socket = str(self.root / "launcher.sock")
        self.apps = self.root / "apps"
        modules = self.root / "modules"
        (modules / "streamlit").mkdir(parents=True)
        (modules / "streamlit" / "__init__.py").touch()
        (modules / "streamlit" / "__main__.py").write_text(FAKE_STREAMLIT)
        for name, source in (extra_modules or {}).items():
            (modules / name).write_text(source)
        run_sh = self.root / "run.sh"
        run_sh.write_text(FAKE_RUN_SH.format(python=sys.executable))
        run_sh.chmod(0o755)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(modules), os.environ.get("PYTHONPATH")])),
            "APP_LAUNCHER_PRELOAD": preload,
            "APP_LAUNCHER_SOCKET": self.socket,
            "APP_LAUNCHER_STOP_TIMEOUT_S": "1",
            "APP_LAUNCHER_LOG_LEVEL": "info",
            "APP_LAUNCHER_RUN_SH": str(run_sh),
            "GREETING": "from the container",
            "INHERITED": "yes",
        }
        self.wrapped = bool(wrapper)
        with open(self.root / "stdout.log", "wb") as stdout, open(self.root / "stderr.log", "wb") as stderr:
            self.process = subprocess.Popen([*wrapper, sys.executable, "-m", "arduino.app_tools.app_launcher"], env=env, stdout=stdout, stderr=stderr)

    def wait_ready(self):
        wait_until(self.answers, details=lambda: (self.root / "stderr.log").read_text())
        return self

    def answers(self):
        try:
            return self.request({"cmd": "ping"})["ok"]
        except OSError:
            return False

    def request(self, message):
        return client.request(message, self.socket, timeout_s=10)

    def output(self, stream="stdout", until=""):
        """Waits for what the launcher and its apps wrote to the stream to hold `until` and returns it."""
        log = self.root / f"{stream}.log"
        return wait_until(lambda: until in (text := log.read_text()) and text, details=lambda: log.read_text())

    def app(self, name, app_yaml="", main=FAKE_MAIN):
        app = self.apps / name
        (app / "python").mkdir(parents=True)
        (app / "python" / "main.py").write_text(main)
        (app / "app.yaml").write_text(app_yaml)
        return app

    def child(self):
        return self.request({"cmd": "status"})["child"]

    def close(self):
        if self.process.poll() is None:
            os.kill(self.launcher_pid(), signal.SIGTERM)
        return self.process.wait(5)

    def launcher_pid(self):
        """The launcher process as seen from the test: the child of the namespace wrapper, when there is one."""
        if not self.wrapped:
            return self.process.pid
        return next(pid for pid in proc_pids() if f"\nPPid:\t{self.process.pid}\n" in read_proc(pid, "status"))

    def remove(self):
        self.close()
        shutil.rmtree(self.root)


@pytest.fixture(scope="module")
def launcher():
    launcher = Launcher().wait_ready()
    yield launcher
    launcher.remove()


def test_runs_an_app_until_stopped(launcher):
    app = launcher.app("hello")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert started["ok"] and started["warm"] and started["runner"] == "python" and not started["prepared"]
    assert set(started["timings"]) == {"stop_ms", "prepare_ms", "fork_ms"}

    status = launcher.request({"cmd": "status"})
    assert status["child"] == {
        "app": str(app),
        "pid": started["pid"],
        "runner": "python",
        "warm": True,
        "started_at": started["started_at"],
        "stopping": False,
    }
    assert status["preload"]["modules"] == ["json"] and status["preload"]["failed"] == []
    # The preload runs once the launcher is idle
    wait_until(lambda: launcher.request({"cmd": "status"})["preload"]["loaded"] == ["json"])
    assert set(status) == {"ok", "child", "preload", "version", "uptime_s"}

    # The app writes to the container stdout and stderr, as the launcher does
    assert f"hello from __main__ {app}/python/main.py from the container yes\n" in launcher.output(until=f"{app}/python/main.py")
    assert f"warning from {app}/python/main.py\n" in launcher.output("stderr", until=f"warning from {app}")
    assert "app-launcher: Ready on" in launcher.output("stderr")

    stopped = launcher.request({"cmd": "stop"})
    assert stopped == {"ok": True, "stopped": [{"app": str(app), "pid": started["pid"], "exit_code": None, "signal": signal.SIGTERM}]}
    assert gone(started["pid"])
    assert launcher.child() is None
    assert launcher.request({"cmd": "stop"}) == {"ok": True, "stopped": []}


def test_the_request_env_is_laid_over_the_container_one(launcher):
    app = launcher.app("greeting")
    started = launcher.request({"cmd": "start", "app": str(app), "env": {"GREETING": "hi"}})
    assert started["ok"]
    assert f"hello from __main__ {app}/python/main.py hi yes\n" in launcher.output(until=f"{app}/python/main.py")
    launcher.request({"cmd": "stop"})


def test_a_start_stops_the_running_app_first(launcher):
    app = launcher.app("again")
    first = launcher.request({"cmd": "start", "app": str(app)})
    second = launcher.request({"cmd": "start", "app": str(app)})
    assert second["ok"] and second["pid"] != first["pid"] and second["timings"]["stop_ms"] > 0
    assert gone(first["pid"])
    assert launcher.child()["pid"] == second["pid"]

    third = launcher.request({"cmd": "start", "app": str(launcher.app("other"))})
    assert third["ok"] and gone(second["pid"])
    assert launcher.request({"cmd": "stop"})["stopped"][0]["pid"] == third["pid"]


def test_an_app_dying_on_its_own_is_forgotten(launcher):
    started = launcher.request({"cmd": "start", "app": str(launcher.app("crashing"))})
    os.kill(started["pid"], signal.SIGKILL)
    assert wait_until(lambda: launcher.child() is None)


def test_off_a_container_a_stop_kills_the_app_process_group_only(launcher):
    marker = f"lingering-{launcher.root.name}"
    app = launcher.app("lingering-here", main=LINGERING_MAIN)
    started = launcher.request({"cmd": "start", "app": str(app), "env": {"MARKER": marker}})
    launcher.output(until=f"helper started {marker}")
    launcher.request({"cmd": "stop"})
    assert gone(started["pid"])
    # Without a container of its own, the launcher cannot tell the helper from the processes of the test
    helpers = tagged(marker)
    assert helpers, "the helper outside the app process group is left alone"
    for pid in helpers:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(not can_unshare(), reason="needs Linux and unprivileged user and pid namespaces")
def test_a_stop_kills_what_the_app_left_behind():
    contained = Launcher(wrapper=UNSHARE).wait_ready()
    try:
        marker = f"lingering-{contained.root.name}"
        app = contained.app("lingering", main=LINGERING_MAIN)
        started = contained.request({"cmd": "start", "app": str(app), "env": {"MARKER": marker}})
        contained.output(until=f"helper started {marker}")
        assert tagged(marker), "the helper runs"

        stopped = contained.request({"cmd": "stop"})
        assert stopped["stopped"][0]["pid"] == started["pid"]
        assert wait_until(lambda: not tagged(marker)), "the helper, in a session of its own, outlived the app"
        assert "Killed what was left behind" in contained.output("stderr")
        assert contained.answers(), "the launcher survives its own sweep"
    finally:
        contained.remove()


def test_runs_streamlit_apps_through_streamlit_on_port_7000(launcher):
    app = launcher.app("dashboard", app_yaml="bricks:\n  - arduino:streamlit_ui: {}\n")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert (started["warm"], started["runner"], started["prepared"]) == (False, "streamlit", True)
    launcher.output(until=f"streamlit run --server.port 7000 {app}/python/main.py {app}\n")
    assert launcher.request({"cmd": "stop"})["stopped"][0]["signal"] == signal.SIGTERM


def test_prepares_an_app_with_dependencies_when_they_change(launcher):
    app = launcher.app("deps")
    requirements = app / "python" / "requirements.txt"
    requirements.write_text("requests\n")
    first = launcher.request({"cmd": "start", "app": str(app), "env": {"GREETING": "deps"}})
    assert (first["ok"], first["prepared"], first["warm"]) == (True, True, False)
    assert first["timings"]["prepare_ms"] > 0
    # The preparation runs with the app environment, its output on the container stdout ahead of the app's
    output = launcher.output(until=f"hello from __main__ {app}/python/main.py deps yes\n")
    assert f"installing {requirements}\nprepared with GREETING=deps INHERITED=yes\n" in output
    assert output.index("prepared with GREETING=deps") < output.index(f"hello from __main__ {app}")
    assert (app / ".cache" / "prepared.sha").exists()

    second = launcher.request({"cmd": "start", "app": str(app)})
    assert (second["ok"], second["prepared"], second["warm"]) == (True, False, False)
    assert (app / "prepared.marker").read_text() == "prepared\n"

    requirements.write_text("requests==2.32.0\n")
    assert launcher.request({"cmd": "start", "app": str(app)})["prepared"]
    assert (app / "prepared.marker").read_text() == "prepared\nprepared\n"
    launcher.request({"cmd": "stop"})


def test_a_failing_preparation_starts_nothing(launcher):
    app = launcher.app("broken")
    (app / "python" / "requirements.txt").write_text("missing-package\n")
    failed = launcher.request({"cmd": "start", "app": str(app)})
    assert failed["error"]["code"] == "prepare_failed"
    assert failed["error"]["message"].endswith(f":\ninstalling {app}/python/requirements.txt\nNo solution found for missing-package")
    assert launcher.child() is None
    assert not (app / ".cache" / "prepared.sha").exists()


def test_refuses_what_is_not_an_app(launcher):
    assert launcher.request({"cmd": "start", "app": "relative/app"})["error"]["code"] == "bad_request"
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "missing")})["error"]["code"] == "not_found"
    (launcher.apps / "empty").mkdir()
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "empty")})["error"]["code"] == "not_found"
    refused = launcher.request({"cmd": "start", "app": str(launcher.app("bad-env")), "env": {"GREETING": 1}})
    assert refused["error"]["code"] == "bad_request"
    assert launcher.child() is None


def test_the_client_prints_the_response(launcher, capsys):
    app = launcher.app("client")
    assert client.main(["--socket", launcher.socket, "start", str(app), "--env", "GREETING=a=b"]) == 0
    pid = json.loads(capsys.readouterr().out)["pid"]
    launcher.output(until=f"hello from __main__ {app}/python/main.py a=b yes\n")
    assert client.main(["--socket", launcher.socket, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["child"]["pid"] == pid
    assert client.main(["--socket", launcher.socket, "start", str(launcher.apps / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "not_found"
    assert client.main(["--socket", launcher.socket, "stop"]) == 0
    assert json.loads(capsys.readouterr().out)["stopped"][0]["pid"] == pid


# Last: it stops the shared launcher
def test_exits_on_sigterm_stopping_its_app(launcher):
    started = launcher.request({"cmd": "start", "app": str(launcher.app("last"))})
    assert launcher.close() == 0
    assert gone(started["pid"])
    assert not os.path.exists(launcher.socket)


def test_a_start_does_not_wait_for_the_preload():
    launcher = Launcher(preload="slow", extra_modules={"slow.py": "import time\ntime.sleep(30)\n"}).wait_ready()
    try:
        app = launcher.app("early")
        started = launcher.request({"cmd": "start", "app": str(app)})
        assert started["ok"], started
        assert launcher.request({"cmd": "status"})["preload"]["loaded"] == []
        launcher.output(until="hello from __main__")
    finally:
        launcher.remove()


def test_exits_on_sigterm_during_the_preload():
    launcher = Launcher(preload="slow", extra_modules={"slow.py": "import time\nprint('importing', flush=True)\ntime.sleep(30)\n"})
    try:
        launcher.output(until="importing")
        launcher.process.send_signal(signal.SIGTERM)
        assert launcher.process.wait(5) == 0
    finally:
        launcher.remove()
