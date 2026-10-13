# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_tools.app_launcher.applog import AppLog


def read(path):
    return path.read_bytes() if path.exists() else None


def test_appends_to_an_existing_log_and_creates_its_folder(tmp_path):
    path = tmp_path / ".cache" / "logs" / "app.log"
    log = AppLog(str(path))
    log.write(b"first\n")
    log.close()
    log = AppLog(str(path))
    log.write(b"second\n")
    log.close()
    assert read(path) == b"first\nsecond\n"


def test_rotates_into_a_single_previous_file(tmp_path):
    path = tmp_path / "app.log"
    log = AppLog(str(path), max_bytes=10)
    log.write(b"aaaaaa")
    log.write(b"bbb")
    assert (read(path), read(tmp_path / "app.log.1")) == (b"aaaaaabbb", None)
    log.write(b"cc")
    assert (read(path), read(tmp_path / "app.log.1")) == (b"cc", b"aaaaaabbb")
    log.write(b"dddddddddddd")
    log.close()
    assert (read(path), read(tmp_path / "app.log.1")) == (b"dddddddddddd", b"cc")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["app.log", "app.log.1"]
