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

from arduino.app_tools.app_launcher import client, confinement
from arduino.app_tools.app_launcher.supervisor import _descends_from

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher forks, POSIX only")

FAKE_MAIN = """
import os, signal, sys
print("hello from", __name__, sys.argv[0], os.environ["GREETING"], os.environ["INHERITED"], flush=True)
signal.pause()
"""

# An app that tries to drive the launcher: it sends a stop through the client, as a library of its own could
ROGUE_MAIN = """
import os, signal, subprocess, sys
client = [sys.executable, "-m", "arduino.app_tools.app_launcher", "--socket", os.environ["APP_LAUNCHER_SOCKET"], "stop"]
done = subprocess.run(client, capture_output=True, text=True)
print("rogue got:", (done.stdout + done.stderr).strip(), flush=True)
signal.pause()
"""

# An app that leaves a helper behind in a session of its own, out of the app's process group
LINGERING_MAIN = """
import signal, subprocess, sys
helper = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"], start_new_session=True)
print("helper", helper.pid, flush=True)
signal.pause()
"""

# Apps that plant a link where the launcher writes, then exit: the next start must not write through it
LOG_PLANTING_MAIN = """
import os
os.remove(".cache/logs/app.log")
os.symlink(os.environ["VICTIM"], ".cache/logs/app.log")
"""
HOME_PLANTING_MAIN = """
import os, shutil
shutil.rmtree(".cache/home")
os.symlink("/", ".cache/home")
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
echo "prepared at home $HOME with PLANTED=${{PLANTED:-nothing}}"
"""


def landlock_abi():
    try:
        return confinement.abi()
    except confinement.Unavailable:
        return None


LANDLOCK = landlock_abi()

# An app that peeks at a sibling app and the launcher home, binds its ports and reports what happened
PEEKING_MAIN = """
import os, signal, socket, sys, tempfile
def attempt(name, fn):
    try:
        fn(); print(name + ": allowed", flush=True)
    except OSError as e:
        print(name + ": denied " + e.strerror, flush=True)
sibling = os.path.join(os.path.dirname(os.getcwd()), "hello")
print("home:", os.environ["HOME"], flush=True)
attempt("read sibling", lambda: open(os.path.join(sibling, "app.yaml")).read())
attempt("list apps", lambda: os.listdir(os.path.dirname(os.getcwd())))
attempt("read own", lambda: open("app.yaml").read())
attempt("write tmp", lambda: tempfile.TemporaryFile().close())
attempt("write home", lambda: open(os.path.join(os.environ["HOME"], "note"), "w").close())
attempt("write launcher home", lambda: open(os.path.join(os.environ["LAUNCHER_HOME"], "note"), "w").close())
attempt("bind 7000", lambda: socket.socket().bind(("127.0.0.1", 7000)))
attempt("bind 7001", lambda: socket.socket().bind(("127.0.0.1", 7001)))
attempt("bind ephemeral", lambda: socket.socket().bind(("127.0.0.1", 0)))
signal.pause()
"""


def wait_until(condition, timeout=5.0, details=lambda: ""):
    deadline = time.monotonic() + timeout
    while not (result := condition()):
        assert time.monotonic() < deadline, f"timed out {details()}"
        time.sleep(0.02)
    return result


def short_folder(prefix="launcher", parent="/tmp"):
    # AF_UNIX paths are limited to about 100 bytes
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()


def spawn(root, preload="json", confinement=None, home=None):
    modules = root / "modules"
    (modules / "streamlit").mkdir(parents=True, exist_ok=True)
    (modules / "streamlit" / "__init__.py").touch()
    (modules / "streamlit" / "__main__.py").write_text(FAKE_STREAMLIT)
    run_sh = root / "run.sh"
    run_sh.write_text(FAKE_RUN_SH.format(python=sys.executable))
    run_sh.chmod(0o755)
    # As in the image: the alias is a link to the link the launcher manages
    if not (root / "app").is_symlink():
        (root / "app").symlink_to(root / "current-app")
    home = str(home or Path.home())
    env = {
        **os.environ,
        "HOME": home,
        "LAUNCHER_HOME": home,
        "PYTHONPATH": str(modules),
        "APP_LAUNCHER_PRELOAD": preload,
        "APP_LAUNCHER_SOCKET": str(root / "launcher.sock"),
        "APP_LAUNCHER_STOP_TIMEOUT_S": "1",
        "APP_LAUNCHER_RUN_SH": str(run_sh),
        "APP_LAUNCHER_APP_ALIAS": str(root / "app"),
        # Confined for real where the kernel allows it, which the test apps under /tmp survive; off on macOS
        "APP_LAUNCHER_CONFINEMENT": confinement or ("required" if LANDLOCK else "off"),
        "GREETING": "from the launcher",
        "INHERITED": "yes",
    }
    output = open(root / "launcher.log", "wb")
    process = subprocess.Popen([sys.executable, "-m", "arduino.app_tools.app_launcher"], cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    output.close()
    return process


class Launcher:
    def __init__(self, root: Path, confinement=None):
        self.root = root
        self.socket = str(root / "launcher.sock")
        self.app_link = root / "current-app"
        # The apps and the launcher home live outside the folders a confined app may reach, /tmp among them
        self.apps = short_folder("apps", "/var/tmp")
        self.home = short_folder("home", "/var/tmp")
        self.process = spawn(root, confinement=confinement, home=self.home)
        wait_until(self.answers, details=lambda: (root / "launcher.log").read_text())

    def answers(self):
        try:
            return self.request({"cmd": "ping"})["ok"]
        except OSError:
            return False

    def request(self, message):
        return client.request(message, self.socket, timeout_s=10)

    def app(self, name, app_yaml=""):
        app = self.apps / name
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
    shutil.rmtree(launcher.apps)
    shutil.rmtree(launcher.home)


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def log_of(app, until=""):
    """Waits for the app log to hold `until` and returns it."""
    log = app / ".cache" / "logs" / "app.log"

    def written():
        text = log.read_text() if log.exists() else ""
        return text if text and until in text else None

    return wait_until(written)


def test_runs_an_app_until_stopped(launcher):
    app = launcher.app("hello")
    started = launcher.request({"cmd": "start", "app": str(app), "env": {"GREETING": "hi"}, "mode": "replace"})
    assert started["ok"] and started["warm"] and started["runner"] == "python" and not started["prepared"]
    assert set(started["timings"]) == {"stop_ms", "prepare_ms", "fork_ms"}

    status = launcher.request({"cmd": "status"})
    child = {"app": str(app), "pid": started["pid"], "state": "running", "started_at": started["started_at"], "runner": "python", "warm": True}
    assert status["children"] == [child]
    assert status["preload"] == {"modules": ["json"], "failed": []}

    assert log_of(app) == f"hello from __main__ {app}/python/main.py hi yes\n"
    assert os.readlink(launcher.app_link) == str(app)
    assert (launcher.root / "app" / "python" / "main.py").read_text() == FAKE_MAIN, "the alias reaches the app"

    stopped = launcher.request({"cmd": "stop", "app": str(app)})
    assert stopped == {"ok": True, "stopped": [{"app": str(app), "pid": started["pid"], "exit_code": None, "signal": signal.SIGTERM}]}
    assert gone(started["pid"])
    assert launcher.request({"cmd": "status"})["children"] == []
    assert not launcher.app_link.is_symlink()
    assert launcher.request({"cmd": "stop", "app": str(app)}) == {"ok": True, "stopped": []}


def test_replace_stops_the_running_app_first(launcher):
    app = launcher.app("again")
    first = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    link = os.lstat(launcher.app_link)
    second = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    assert second["ok"] and second["pid"] != first["pid"]
    assert gone(first["pid"])
    assert [child["pid"] for child in launcher.request({"cmd": "status"})["children"]] == [second["pid"]]
    assert os.lstat(launcher.app_link) == link, "a restart of the same app keeps the link in place"

    other = launcher.app("other")
    third = launcher.request({"cmd": "start", "app": str(other), "mode": "replace"})
    assert third["ok"] and gone(second["pid"])
    assert os.readlink(launcher.app_link) == str(other)

    refused = launcher.request({"cmd": "start", "app": str(app), "mode": "add"})
    assert refused["error"]["code"] == "unsupported"
    assert [child["pid"] for child in launcher.request({"cmd": "status"})["children"]] == [third["pid"]]

    assert launcher.request({"cmd": "stop"})["stopped"][0]["pid"] == third["pid"]


def test_the_alias_goes_away_when_the_last_app_dies_on_its_own(launcher):
    app = launcher.app("crashing")
    started = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    assert os.readlink(launcher.app_link) == str(app)
    os.kill(started["pid"], signal.SIGKILL)
    assert wait_until(lambda: launcher.request({"cmd": "status"})["children"] == [])
    assert wait_until(lambda: not launcher.app_link.is_symlink())


def test_a_link_planted_as_the_log_carries_no_write(launcher):
    victim = launcher.root / "precious.txt"
    victim.write_text("precious\n")
    app = launcher.app("log-planting")
    (app / "python" / "main.py").write_text(LOG_PLANTING_MAIN)
    assert launcher.request({"cmd": "start", "app": str(app), "env": {"VICTIM": str(victim)}})["ok"]
    assert wait_until(lambda: launcher.request({"cmd": "status"})["children"] == [])
    assert (app / ".cache" / "logs" / "app.log").is_symlink()

    assert launcher.request({"cmd": "start", "app": str(app), "env": {"VICTIM": str(victim)}})["ok"], "the app runs, its output dropped"
    assert wait_until(lambda: launcher.request({"cmd": "status"})["children"] == [])
    assert victim.read_text() == "precious\n"
    assert "Cannot write the log" in (launcher.root / "launcher.log").read_text()


def test_a_link_planted_as_the_home_refuses_the_start(launcher):
    app = launcher.app("home-planting")
    (app / "python" / "main.py").write_text(HOME_PLANTING_MAIN)
    assert launcher.request({"cmd": "start", "app": str(app)})["ok"]
    assert wait_until(lambda: launcher.request({"cmd": "status"})["children"] == [])
    assert (app / ".cache" / "home").is_symlink()

    refused = launcher.request({"cmd": "start", "app": str(app)})
    assert refused["error"]["code"] == "internal" and "home" in refused["error"]["message"]
    assert launcher.request({"cmd": "status"})["children"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="the launcher walks /proc, Linux only")
def test_a_stop_kills_what_the_app_left_behind(launcher):
    app = launcher.app("lingering")
    (app / "python" / "main.py").write_text(LINGERING_MAIN)
    started = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    helper = int(log_of(app, until="helper").split()[1])
    assert not gone(helper)

    stopped = launcher.request({"cmd": "stop"})
    assert stopped["stopped"][0]["pid"] == started["pid"]
    assert wait_until(lambda: gone(helper)), "the helper, in a session of its own, outlived the app"


@pytest.mark.skipif(not hasattr(socket, "SO_PEERCRED"), reason="the peer pid exists on Linux only")
def test_refuses_requests_from_the_processes_of_an_app(launcher):
    app = launcher.app("rogue")
    (app / "python" / "main.py").write_text(ROGUE_MAIN)
    started = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
    assert started["ok"]
    assert '"forbidden"' in log_of(app, until="rogue got:")
    assert [child["pid"] for child in launcher.request({"cmd": "status"})["children"]] == [started["pid"]], "the app kept running"
    assert launcher.request({"cmd": "stop"})["stopped"][0]["pid"] == started["pid"], "the test, outside, still can"


@pytest.mark.skipif(LANDLOCK is None, reason="the kernel has no Landlock")
def test_confines_an_app_to_its_own_folder_and_ports(launcher):
    if not (launcher.apps / "hello").exists():
        launcher.app("hello")
    app = launcher.app("peeking")
    (app / "python" / "main.py").write_text(PEEKING_MAIN)
    started = launcher.request({"cmd": "start", "app": str(app), "mode": "replace", "ports": [7000]})
    assert started["ok"] and started["confined"]
    log = log_of(app, until="bind ephemeral")
    assert f"home: {app}/.cache/home" in log
    assert "read sibling: denied" in log and "list apps: denied" in log
    assert "read own: allowed" in log and "write tmp: allowed" in log
    assert "write home: allowed" in log and "write launcher home: denied" in log
    assert "bind 7000: allowed" in log and "bind 7001: denied" in log and "bind ephemeral: allowed" in log
    assert launcher.request({"cmd": "status"})["confinement"] == {"required": True, "landlock_abi": LANDLOCK}
    launcher.request({"cmd": "stop"})


@pytest.mark.skipif(LANDLOCK is not None, reason="this kernel has Landlock: the launcher confines for real")
def test_starts_nothing_when_confinement_is_required_but_the_kernel_has_no_landlock():
    root = short_folder()
    launcher = Launcher(root, confinement="required")
    try:
        app = launcher.app("hello")
        refused = launcher.request({"cmd": "start", "app": str(app), "mode": "replace"})
        assert refused["error"]["code"] == "unsupported" and "Landlock" in refused["error"]["message"]
        assert launcher.request({"cmd": "status"})["confinement"] == {"required": True, "landlock_abi": None}
        assert "no Landlock" in (root / "launcher.log").read_text()
    finally:
        launcher.close()
        shutil.rmtree(root)
        shutil.rmtree(launcher.apps)
        shutil.rmtree(launcher.home)


def test_runs_streamlit_apps_through_streamlit_on_port_7000(launcher):
    app = launcher.app("dashboard", app_yaml="bricks:\n  - arduino:streamlit_ui: {}\n")
    started = launcher.request({"cmd": "start", "app": str(app)})
    assert (started["warm"], started["runner"], started["prepared"]) == (False, "streamlit", True)
    streamlit = f"streamlit run --server.port 7000 {app}/python/main.py {app}\n"
    assert log_of(app, until=streamlit).endswith(streamlit)
    assert launcher.request({"cmd": "stop"})["stopped"][0]["signal"] == signal.SIGTERM


def test_prepares_an_app_with_dependencies_when_they_change(launcher):
    app = launcher.app("deps")
    requirements = app / "python" / "requirements.txt"
    requirements.write_text("requests\n")
    # The request environment reaches the app, never the preparation, which runs before any confinement
    first = launcher.request({"cmd": "start", "app": str(app), "env": {"PLANTED": "by the request"}})
    assert (first["ok"], first["prepared"], first["warm"]) == (True, True, False)
    assert first["timings"]["prepare_ms"] > 0
    assert log_of(app, until="hello from").startswith(
        f"installing {requirements}\nprepared at home {app}/.cache/home with PLANTED=nothing\nhello from __main__ {app}/python/main.py"
    )
    assert (app / ".cache" / "launcher" / "prepared.sha").exists()

    second = launcher.request({"cmd": "start", "app": str(app)})
    assert (second["ok"], second["prepared"], second["warm"]) == (True, False, False)
    assert (app / "prepared.marker").read_text() == "prepared\n"

    requirements.write_text("requests==2.32.0\n")
    assert launcher.request({"cmd": "start", "app": str(app)})["prepared"]
    assert (app / "prepared.marker").read_text() == "prepared\nprepared\n"
    launcher.request({"cmd": "stop"})


@pytest.mark.skipif(LANDLOCK is None, reason="the preparation runs through the wrapper only when confined")
def test_the_preparation_wrapper_imports_nothing_from_where_the_launcher_runs(launcher):
    # A package planted in the launcher's working directory, named as the library: the wrapper must not pick it up
    planted = launcher.root / "arduino"
    planted.mkdir()
    (planted / "__init__.py").write_text("raise RuntimeError('planted package imported')\n")
    app = launcher.app("wrapped")
    (app / "python" / "requirements.txt").write_text("requests\n")
    try:
        started = launcher.request({"cmd": "start", "app": str(app)})
    finally:
        shutil.rmtree(planted)
    assert started["ok"] and started["prepared"], started
    launcher.request({"cmd": "stop"})


def test_a_failing_preparation_starts_nothing(launcher):
    app = launcher.app("broken")
    (app / "python" / "requirements.txt").write_text("missing-package\n")
    failed = launcher.request({"cmd": "start", "app": str(app)})
    assert failed["error"]["code"] == "prepare_failed"
    assert failed["error"]["message"].endswith(":\ninstalling " + f"{app}/python/requirements.txt\nNo solution found for missing-package")
    assert launcher.request({"cmd": "status"})["children"] == []
    assert not (app / ".cache" / "launcher" / "prepared.sha").exists()
    assert log_of(app).endswith("No solution found for missing-package\n")


def test_refuses_what_is_not_an_app(launcher):
    assert launcher.request({"cmd": "start", "app": "relative/app"})["error"]["code"] == "bad_request"
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "missing")})["error"]["code"] == "not_found"
    (launcher.apps / "empty").mkdir()
    assert launcher.request({"cmd": "start", "app": str(launcher.apps / "empty")})["error"]["code"] == "not_found"


def test_a_peer_gone_before_the_check_counts_as_launched():
    gone = next(pid for pid in range(2**22 - 1, 2**22 - 10000, -1) if not os.path.exists(f"/proc/{pid}"))
    assert _descends_from(gone, os.getpid()), "nothing proves it was not one of ours"
    assert not _descends_from(gone, os.getpid(), vanished=False), "a sweep kills only what still exists"
    assert _descends_from(os.getpid(), os.getpid())


def test_the_client_prints_the_response(launcher, capsys):
    app = launcher.app("client")
    assert client.main(["--socket", launcher.socket, "start", str(app), "--env", "GREETING=a=b"]) == 0
    pid = json.loads(capsys.readouterr().out)["pid"]
    assert log_of(app).endswith(" a=b yes\n")
    assert client.main(["--socket", launcher.socket, "status"]) == 0
    assert [child["pid"] for child in json.loads(capsys.readouterr().out)["children"]] == [pid]
    assert client.main(["--socket", launcher.socket, "start", str(launcher.apps / "missing")]) == 1
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
