# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import shutil
import signal
import socket
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
print("hello from", __name__, sys.argv[0], os.environ["ARDUINO_APP_HTTP_PORT"], os.environ["GREETING"], os.environ["INHERITED"], flush=True)
signal.pause()
"""

# Stands in for Streamlit: prints what `python -m streamlit` got
FAKE_STREAMLIT = """
import os, signal, sys
print("streamlit", *sys.argv[1:], os.getcwd(), flush=True)
signal.pause()
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(condition, timeout=5.0, details=lambda: ""):
    deadline = time.monotonic() + timeout
    while not (result := condition()):
        assert time.monotonic() < deadline, f"timed out {details()}"
        time.sleep(0.02)
    return result


def short_folder():
    # AF_UNIX paths are limited to about 100 bytes
    return Path(tempfile.mkdtemp(prefix="launcher", dir="/tmp")).resolve()


def spawn(root, preload="json", port=None):
    modules = root / "modules"
    (modules / "streamlit").mkdir(parents=True, exist_ok=True)
    (modules / "streamlit" / "__init__.py").touch()
    (modules / "streamlit" / "__main__.py").write_text(FAKE_STREAMLIT)
    (root / "launcher").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(modules),
        "APP_LAUNCHER_PRELOAD": preload,
        "APP_LAUNCHER_SOCKET": str(root / "launcher.sock"),
        "APP_LAUNCHER_LINK": str(root / "launcher" / "current"),
        "APP_LAUNCHER_PORTS": str(port or free_port()),
        "APP_LAUNCHER_STOP_TIMEOUT_S": "1",
        "GREETING": "from the launcher",
        "INHERITED": "yes",
    }
    output = open(root / "launcher.log", "wb")
    process = subprocess.Popen([sys.executable, "-m", "arduino.app_tools.app_launcher"], env=env, stdout=output, stderr=subprocess.STDOUT)
    output.close()
    return process


class Launcher:
    def __init__(self, root: Path):
        self.root = root
        self.socket = str(root / "launcher.sock")
        self.port = free_port()
        self.link = root / "launcher" / "current"
        self.process = spawn(root, port=self.port)
        wait_until(self.answers, details=lambda: (root / "launcher.log").read_text())

    def answers(self):
        try:
            return self.request({"cmd": "ping"})["ok"]
        except OSError:
            return False

    def request(self, message):
        return client.request(message, self.socket, timeout_s=10)

    def app(self, name, app_yaml=""):
        app = self.root / name
        (app / "python").mkdir(parents=True)
        (app / "python" / "main.py").write_text(FAKE_MAIN)
        (app / "app.yaml").write_text(app_yaml)
        return app

    def close(self):
        self.process.send_signal(signal.SIGTERM)
        return self.process.wait(5)


@pytest.fixture(scope="module")
def launcher():
    root = short_folder()
    launcher = Launcher(root)
    yield launcher
    if launcher.process.poll() is None:
        launcher.close()
    shutil.rmtree(root)


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def log_of(app):
    log = app / ".cache" / "logs" / "app.log"
    return wait_until(lambda: log.exists() and log.read_text())


def test_runs_an_app_until_stopped(launcher):
    app = launcher.app("hello")
    started = launcher.request({"cmd": "start", "app": str(app), "env": {"GREETING": "hi"}, "mode": "replace"})
    assert started["ok"] and started["warm"] and started["runner"] == "python"
    assert set(started["timings"]) == {"stop_ms", "fork_ms"}

    status = launcher.request({"cmd": "status"})
    child = {"app": str(app), "pid": started["pid"], "state": "running", "started_at": started["started_at"], "runner": "python", "warm": True}
    assert status["children"] == [{**child, "http_port": launcher.port + 10000}]
    assert status["preload"] == {"modules": ["json"], "failed": []}
    assert os.readlink(launcher.link) == str(app)

    assert log_of(app) == f"hello from __main__ {app}/python/main.py {launcher.port + 10000} hi yes\n"

    stopped = launcher.request({"cmd": "stop", "app": str(app)})
    assert stopped == {"ok": True, "stopped": [{"app": str(app), "pid": started["pid"], "exit_code": None, "signal": signal.SIGTERM}]}
    assert gone(started["pid"])
    assert launcher.request({"cmd": "status"})["children"] == []
    assert launcher.request({"cmd": "stop", "app": str(app)}) == {"ok": True, "stopped": []}


def test_replace_stops_the_running_app_first(launcher):
    app = launcher.app("again")
    first = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    second = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    assert second["ok"] and second["pid"] != first["pid"]
    assert gone(first["pid"])
    assert [child["pid"] for child in launcher.request({"cmd": "status"})["children"]] == [second["pid"]]

    refused = launcher.request({"cmd": "start", "app": str(app), "mode": "add"})
    assert refused["error"]["code"] == "unsupported"
    assert [child["pid"] for child in launcher.request({"cmd": "status"})["children"]] == [second["pid"]]

    assert launcher.request({"cmd": "stop"})["stopped"][0]["pid"] == second["pid"]


def test_runs_streamlit_apps_through_streamlit(launcher):
    app = launcher.app("dashboard", app_yaml="bricks:\n  - arduino:streamlit_ui: {}\n")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert (started["warm"], started["runner"]) == (False, "streamlit")
    assert log_of(app) == f"streamlit run --server.port {launcher.port + 10000} {app}/python/main.py {app}\n"
    assert launcher.request({"cmd": "stop"})["stopped"][0]["signal"] == signal.SIGTERM


def test_refuses_what_is_not_an_app(launcher):
    assert launcher.request({"cmd": "start", "app": "relative/app"})["error"]["code"] == "bad_request"
    assert launcher.request({"cmd": "start", "app": str(launcher.root / "missing")})["error"]["code"] == "not_found"
    (launcher.root / "empty").mkdir()
    assert launcher.request({"cmd": "start", "app": str(launcher.root / "empty")})["error"]["code"] == "not_found"


def test_the_client_prints_the_response(launcher, capsys):
    app = launcher.app("client")
    assert client.main(["--socket", launcher.socket, "start", str(app), "--env", "GREETING=a=b"]) == 0
    pid = json.loads(capsys.readouterr().out)["pid"]
    assert log_of(app).endswith(" a=b yes\n")
    assert client.main(["--socket", launcher.socket, "status"]) == 0
    assert [child["pid"] for child in json.loads(capsys.readouterr().out)["children"]] == [pid]
    assert client.main(["--socket", launcher.socket, "start", str(launcher.root / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "not_found"
    assert client.main(["--socket", launcher.socket, "stop", str(app)]) == 0
    assert json.loads(capsys.readouterr().out)["stopped"][0]["pid"] == pid


# Last: it stops the shared launcher
def test_exits_on_sigterm_stopping_its_apps(launcher):
    started = launcher.request({"cmd": "start", "app": str(launcher.app("last"))})
    assert launcher.close() == 0
    assert gone(started["pid"])
    assert not os.path.exists(launcher.socket)


def test_exits_on_sigterm_during_the_preload():
    root = short_folder()
    (root / "modules").mkdir()
    (root / "modules" / "slow.py").write_text("import time\nprint('importing', flush=True)\ntime.sleep(30)\n")
    process = spawn(root, preload="slow")
    wait_until(lambda: "importing" in (root / "launcher.log").read_text())
    process.send_signal(signal.SIGTERM)
    assert process.wait(5) == 0
    shutil.rmtree(root)
