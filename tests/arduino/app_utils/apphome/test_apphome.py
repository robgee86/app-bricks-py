# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

import pytest

from arduino.app_utils.apphome import app_folder


@pytest.fixture
def cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return os.getcwd()


def test_the_app_folder_is_the_working_directory_without_app_home(monkeypatch, cwd):
    monkeypatch.delenv("APP_HOME", raising=False)
    assert app_folder() == cwd
    assert app_folder("data", "arduino.db") == os.path.join(cwd, "data", "arduino.db")


def test_the_app_folder_is_the_working_directory_when_app_home_does_not_exist(monkeypatch, cwd):
    monkeypatch.setenv("APP_HOME", os.path.join(cwd, "missing"))
    assert app_folder() == cwd


def test_the_app_folder_is_the_working_directory_when_app_home_is_empty(monkeypatch, cwd):
    monkeypatch.setenv("APP_HOME", "")
    assert app_folder() == cwd


def test_the_app_folder_is_app_home_when_it_exists(monkeypatch, tmp_path, cwd):
    home = tmp_path / "app"
    home.mkdir()
    monkeypatch.setenv("APP_HOME", str(home))
    assert app_folder() == str(home)
    assert app_folder("certs") == os.path.join(str(home), "certs")
