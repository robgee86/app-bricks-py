# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import sys

import pytest

from arduino.app_tools.app_launcher.child import Launch, Runner, environment, runner, venv_interpreter


def launch(app, env, **fields):
    return Launch(app=str(app), env=env, http_port=17000, output_fd=-1, **fields)


@pytest.fixture
def venv(tmp_path):
    """An app with an empty venv, as `uv venv` leaves it."""
    site_packages = tmp_path / ".cache" / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_virtualenv.pth").touch()
    (site_packages / "_virtualenv.py").touch()
    python = tmp_path / ".cache" / ".venv" / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    (tmp_path / "python").mkdir()
    return tmp_path, site_packages, str(python)


def test_an_app_without_dependencies_of_its_own_runs_warm(venv):
    app, _, _ = venv
    (app / "python" / "requirements.txt").write_text("\n  \n")
    assert venv_interpreter(str(app)) is None


@pytest.mark.parametrize(
    "dependency",
    [
        lambda app, site: (site / "numpy-2.3.0.dist-info").mkdir(),
        lambda app, site: (site / "editable.pth").touch(),
        lambda app, site: (app / "python" / "requirements.txt").write_text("requests\n"),
        lambda app, site: (app / "python-libraries").mkdir() or (app / "python-libraries" / "lib-1.0-py3-none-any.whl.installed").touch(),
    ],
)
def test_an_app_with_dependencies_of_its_own_runs_its_venv(venv, dependency):
    app, site_packages, python = venv
    dependency(app, site_packages)
    assert venv_interpreter(str(app)) == python


def test_an_app_without_venv_runs_warm(tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "requirements.txt").write_text("requests\n")
    assert venv_interpreter(str(tmp_path)) is None


def test_streamlit_apps_are_exec_d_through_streamlit(tmp_path):
    assert runner(str(tmp_path)) is Runner.PYTHON
    (tmp_path / "app.yaml").write_text("bricks:\n  - arduino:streamlit_ui: {}\n")
    assert runner(str(tmp_path)) is Runner.STREAMLIT
    main_py = str(tmp_path / "python" / "main.py")
    assert launch(tmp_path, {}, runner=Runner.STREAMLIT).exec_argv == [sys.executable, "-m", "streamlit", "run", "--server.port", "17000", main_py]
    assert launch(tmp_path, {}, runner=Runner.STREAMLIT, interpreter="/venv/bin/python").exec_argv[0] == "/venv/bin/python"
    assert launch(tmp_path, {}, interpreter="/venv/bin/python").exec_argv == ["/venv/bin/python", main_py]
    assert launch(tmp_path, {}).exec_argv is None


def test_environment_is_the_launch_one_plus_the_port(tmp_path):
    assert environment(launch(tmp_path, {"HOME": "/home/app"})) == {"HOME": "/home/app", "ARDUINO_APP_HTTP_PORT": "17000", "PYTHONUNBUFFERED": "1"}


def test_environment_adds_the_app_bricks_to_pythonpath(tmp_path):
    (tmp_path / "bricks").mkdir()
    env = environment(launch(tmp_path, {"PYTHONPATH": "/lib"}))
    assert env["PYTHONPATH"] == os.pathsep.join(["/lib", str(tmp_path / "bricks")])


def test_environment_activates_the_app_venv(tmp_path):
    python = str(tmp_path / ".cache" / ".venv" / "bin" / "python")
    env = environment(launch(tmp_path, {"PATH": "/usr/bin"}, interpreter=python))
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".cache" / ".venv")
    assert env["PATH"] == os.pathsep.join([os.path.dirname(python), "/usr/bin"])
