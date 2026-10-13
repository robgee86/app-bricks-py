# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

from arduino.app_utils.apphome import app_folder


def test_the_app_folder_is_app_without_app_home(monkeypatch):
    monkeypatch.delenv("APP_HOME", raising=False)
    assert app_folder() == "/app"
    assert app_folder("data", "arduino.db") == "/app/data/arduino.db"


def test_the_app_folder_is_app_when_app_home_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_HOME", str(tmp_path / "missing"))
    assert app_folder() == "/app"


def test_the_app_folder_is_app_when_app_home_is_empty(monkeypatch):
    monkeypatch.setenv("APP_HOME", "")
    assert app_folder() == "/app"


def test_the_app_folder_is_app_home_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_HOME", str(tmp_path))
    assert app_folder() == str(tmp_path)
    assert app_folder("certs") == os.path.join(str(tmp_path), "certs")
