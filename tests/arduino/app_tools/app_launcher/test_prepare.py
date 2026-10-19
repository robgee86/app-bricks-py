# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import os
import sys

import pytest

from arduino.app_tools.app_launcher import prepare
from arduino.app_tools.app_launcher.applog import AppLog
from arduino.app_tools.app_launcher.protocol import ErrorCode, LauncherError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher spawns run.sh, POSIX only")


def write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def venv(app):
    write(app / ".cache" / ".venv" / "bin" / "python")


def test_an_app_without_dependencies_has_no_fingerprint(tmp_path):
    write(tmp_path / "python" / "requirements.txt", "\n  \n")
    write(tmp_path / "bricks" / "mine" / "requirements.txt")
    write(tmp_path / "python-libraries" / "README.md", "no wheels")
    write(tmp_path / "app.yaml", "bricks:\n  - arduino:web_ui: {}\n")
    assert prepare.fingerprint(str(tmp_path)) is None


@pytest.mark.parametrize(
    "dependency",
    [
        ("python/requirements.txt", "requests\n"),
        ("bricks/mine/requirements.txt", "numpy\n"),
        ("python-libraries/lib-1.0-py3-none-any.whl", ""),
        ("python-libraries/lib-1.0-py3-none-any.whl.installed", ""),
        ("app.yaml", "bricks:\n  - arduino:streamlit_ui: {}\n"),
    ],
)
def test_each_kind_of_dependency_gets_a_fingerprint(tmp_path, dependency):
    path, content = dependency
    write(tmp_path / path, content)
    assert prepare.fingerprint(str(tmp_path)) is not None


def test_the_fingerprint_follows_the_requirements(tmp_path):
    requirements = tmp_path / "python" / "requirements.txt"
    write(requirements, "requests\n")
    first = prepare.fingerprint(str(tmp_path))
    assert prepare.fingerprint(str(tmp_path)) == first
    write(requirements, "requests==2.32.0\n")
    second = prepare.fingerprint(str(tmp_path))
    assert second != first
    write(tmp_path / "bricks" / "mine" / "requirements.txt", "numpy\n")
    assert prepare.fingerprint(str(tmp_path)) not in (first, second)


def test_installing_a_wheel_keeps_the_fingerprint(tmp_path):
    wheel = tmp_path / "python-libraries" / "lib-1.0-py3-none-any.whl"
    write(wheel)
    before = prepare.fingerprint(str(tmp_path))
    wheel.rename(f"{wheel}.installed")
    assert prepare.fingerprint(str(tmp_path)) == before
    write(tmp_path / "python-libraries" / "lib-1.1-py3-none-any.whl")
    assert prepare.fingerprint(str(tmp_path)) != before


def test_an_app_is_prepared_until_its_fingerprint_is_recorded_with_a_venv(tmp_path):
    app = str(tmp_path)
    assert prepare.needs_preparation(app, "abc")
    prepare.record(app, "abc")
    assert prepare.needs_preparation(app, "abc"), "the venv is missing"
    venv(tmp_path)
    assert not prepare.needs_preparation(app, "abc")
    assert prepare.needs_preparation(app, "def")
    assert (tmp_path / ".cache" / "prepared.sha").read_text() == "abc\n"


def run_sh(tmp_path, body):
    script = tmp_path / "run.sh"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return str(script)


async def run(app, script, timeout_s=5.0):
    """Runs the preparation with a polling reaper standing in for the supervisor one."""
    loop = asyncio.get_running_loop()
    reapers = []

    def watch(pid):
        exited = loop.create_future()

        async def reap():
            while (reaped := os.waitpid(pid, os.WNOHANG))[0] == 0:
                await asyncio.sleep(0.01)
            exited.set_result(reaped[1])

        reapers.append(loop.create_task(reap()))
        return exited

    log = AppLog(str(app))
    try:
        await prepare.run(str(app), {"PATH": os.environ["PATH"]}, script, timeout_s, log, watch, confine=False)
    finally:
        log.close()
        await asyncio.gather(*reapers)


def test_run_appends_the_output_to_the_app_log(tmp_path):
    script = run_sh(tmp_path, 'echo "preparing $APP_DIR $1"\necho oops >&2')
    asyncio.run(run(tmp_path, script))
    assert (tmp_path / ".cache" / "logs" / "app.log").read_text() == f"preparing {tmp_path} prepare\noops\n"


def test_a_failure_reports_the_last_lines(tmp_path):
    script = run_sh(tmp_path, "for i in $(seq 1 30); do echo line $i; done\nexit 3")
    with pytest.raises(LauncherError) as failed:
        asyncio.run(run(tmp_path, script))
    assert failed.value.code is ErrorCode.PREPARE_FAILED
    assert failed.value.message == f"{script} prepare exited with code 3:\n" + "\n".join(f"line {i}" for i in range(11, 31))


def test_a_slow_preparation_is_killed(tmp_path):
    script = run_sh(tmp_path, "echo installing\nsleep 30")
    # Generous timeout: under a loaded test run the echo must still be read before it fires
    with pytest.raises(LauncherError) as failed:
        asyncio.run(run(tmp_path, script, timeout_s=1.5))
    assert failed.value.message == f"{script} prepare timed out after 1.5 s:\ninstalling"


def test_a_missing_run_sh_fails_the_preparation(tmp_path):
    with pytest.raises(LauncherError) as failed:
        asyncio.run(run(tmp_path, str(tmp_path / "missing.sh")))
    assert failed.value.code is ErrorCode.PREPARE_FAILED
