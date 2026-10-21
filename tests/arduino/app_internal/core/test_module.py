# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sys
import types

from arduino.app_internal.core.module import get_app_config


def test_the_app_config_comes_from_the_app_folder(monkeypatch, tmp_path):
    (tmp_path / "app.yaml").write_text("name: from app home\n")
    monkeypatch.chdir(tmp_path)
    assert get_app_config() == {"name": "from app home"}


def test_the_app_config_falls_back_to_the_folder_of_the_main_script(monkeypatch, tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "app.yaml").write_text("name: next to main\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(tmp_path / "python" / "main.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.chdir(tmp_path / "python")
    assert get_app_config() == {"name": "next to main"}


def test_resolve_address_prefers_the_host_the_runtime_names(monkeypatch):
    from arduino.app_internal.core.module import resolve_address, service_host_variable

    monkeypatch.delenv("LOCAL_DEV", raising=False)
    monkeypatch.delenv("REMOTE_DEV", raising=False)
    assert service_host_variable("ei-obj-detection-runner") == "ARDUINO_SERVICE_HOST_EI_OBJ_DETECTION_RUNNER"
    assert resolve_address("dbstorage-influx") == "dbstorage-influx"
    monkeypatch.setenv("ARDUINO_SERVICE_HOST_DBSTORAGE_INFLUX", "my-app-dbstorage-influx-1")
    assert resolve_address("dbstorage-influx") == "my-app-dbstorage-influx-1"
    monkeypatch.setenv("LOCAL_DEV", "true")
    assert resolve_address("dbstorage-influx") == "127.0.0.1", "development overrides come first"
