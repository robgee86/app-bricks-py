# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

from arduino.app_utils.apphome import app_path, app_root


def test_the_app_root_is_app_without_app_home(monkeypatch):
    monkeypatch.delenv("APP_HOME", raising=False)
    assert app_root() == "/app"
    assert app_path("data", "arduino.db") == "/app/data/arduino.db"


def test_the_app_root_is_app_when_app_home_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_HOME", str(tmp_path / "missing"))
    assert app_root() == "/app"


def test_the_app_root_is_app_when_app_home_is_empty(monkeypatch):
    monkeypatch.setenv("APP_HOME", "")
    assert app_root() == "/app"


def test_the_app_root_is_app_home_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_HOME", str(tmp_path))
    assert app_root() == str(tmp_path)
    assert app_path("certs") == os.path.join(str(tmp_path), "certs")
