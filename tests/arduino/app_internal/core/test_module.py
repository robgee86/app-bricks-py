# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sys
import types

from arduino.app_internal.core.module import get_app_config


def test_the_app_config_comes_from_the_app_folder(monkeypatch, tmp_path):
    (tmp_path / "app.yaml").write_text("name: from app home\n")
    monkeypatch.setenv("APP_HOME", str(tmp_path))
    assert get_app_config() == {"name": "from app home"}


def test_the_app_config_falls_back_to_the_folder_of_the_main_script(monkeypatch, tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "app.yaml").write_text("name: next to main\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(tmp_path / "python" / "main.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.setenv("APP_HOME", str(tmp_path / "missing"))
    assert get_app_config() == {"name": "next to main"}
