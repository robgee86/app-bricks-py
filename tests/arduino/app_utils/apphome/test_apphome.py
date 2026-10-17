# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from arduino.app_utils.apphome import app_folder

MAIN = """
import os
from arduino.app_utils import app_folder

print(app_folder())
os.chdir("/")
print(app_folder("data", "arduino.db"))
"""


@pytest.fixture
def cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return os.getcwd()


def make_app(root: Path) -> Path:
    (root / "python").mkdir(parents=True)
    (root / "python" / "main.py").write_text(MAIN)
    (root / "app.yaml").write_text("")
    return root


def test_the_app_folder_is_the_working_directory_without_an_app_above_it(monkeypatch, cwd):
    monkeypatch.setenv("APP_HOME", os.path.join(cwd, "elsewhere"))
    assert app_folder() == cwd
    assert app_folder("data", "arduino.db") == os.path.join(cwd, "data", "arduino.db")


def test_the_app_folder_is_the_nearest_app_above_the_working_directory(monkeypatch, tmp_path):
    app = make_app(tmp_path / "app")
    inside = app / "python" / "lib"
    inside.mkdir()
    monkeypatch.chdir(inside)
    assert app_folder() == str(app)
    assert app_folder("certs") == str(app / "certs")


def test_the_app_folder_follows_the_running_script_wherever_it_runs_from(tmp_path):
    app = make_app(tmp_path / "app")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {**os.environ, "APP_HOME": str(elsewhere)}
    output = subprocess.run([sys.executable, str(app / "python" / "main.py")], cwd=elsewhere, env=env, capture_output=True, text=True, check=True)
    assert output.stdout.splitlines() == [str(app), str(app / "data" / "arduino.db")]
