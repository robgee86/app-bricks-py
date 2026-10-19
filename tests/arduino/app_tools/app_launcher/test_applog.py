# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_tools.app_launcher.applog import AppLog


def read(path):
    return path.read_bytes() if path.exists() else None


def test_appends_to_an_existing_log_and_creates_its_folder(tmp_path):
    path = tmp_path / ".cache" / "logs" / "app.log"
    log = AppLog(str(tmp_path))
    assert log.path == str(path)
    log.write(b"first\n")
    log.close()
    log = AppLog(str(tmp_path))
    log.write(b"second\n")
    log.close()
    assert read(path) == b"first\nsecond\n"


def test_rotates_into_a_single_previous_file(tmp_path):
    logs = tmp_path / ".cache" / "logs"
    path = logs / "app.log"
    log = AppLog(str(tmp_path), max_bytes=10)
    log.write(b"aaaaaa")
    log.write(b"bbb")
    assert (read(path), read(logs / "app.log.1")) == (b"aaaaaabbb", None)
    log.write(b"cc")
    assert (read(path), read(logs / "app.log.1")) == (b"cc", b"aaaaaabbb")
    log.write(b"dddddddddddd")
    log.close()
    assert (read(path), read(logs / "app.log.1")) == (b"dddddddddddd", b"cc")
    assert sorted(p.name for p in logs.iterdir()) == ["app.log", "app.log.1"]


def test_a_fresh_log_holds_one_run(tmp_path):
    logs = tmp_path / ".cache" / "logs"
    previous = AppLog(str(tmp_path), max_bytes=8)
    previous.write(b"old run\n")
    previous.write(b"rotates\n")  # The first write goes to app.log.1
    previous.close()
    assert (logs / "app.log.1").exists()

    log = AppLog(str(tmp_path), fresh=True)
    log.write(b"new run\n")
    log.close()
    assert (logs / "app.log").read_bytes() == b"new run\n"
    assert not (logs / "app.log.1").exists()


@pytest.mark.parametrize("planted", [".cache", ".cache/logs", ".cache/logs/app.log"])
def test_a_link_planted_on_the_way_to_the_log_is_refused(tmp_path, planted):
    # The app may plant links in its folder; the launcher, not confined, must never write through them
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "app.log").write_bytes(b"precious\n")
    link = tmp_path / planted
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(victim if planted != ".cache/logs/app.log" else victim / "app.log")

    with pytest.raises(OSError):
        AppLog(str(tmp_path), fresh=True)
    assert (victim / "app.log").read_bytes() == b"precious\n"
