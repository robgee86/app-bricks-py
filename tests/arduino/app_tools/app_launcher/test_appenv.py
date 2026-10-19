# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import stat

import pytest

from arduino.app_tools.app_launcher import appenv


def test_reads_back_what_was_written_readable_by_the_owner_only(tmp_path):
    assert appenv.read(str(tmp_path)) == {}, "no file, no variables"
    appenv.write(str(tmp_path), {"AUTH_KEY": "s3cret", "PORT": "7000"})
    file = tmp_path / ".cache" / "environment.json"
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert json.loads(file.read_text()) == {"AUTH_KEY": "s3cret", "PORT": "7000"}
    assert appenv.read(str(tmp_path)) == {"AUTH_KEY": "s3cret", "PORT": "7000"}
    appenv.write(str(tmp_path), {"OTHER": "x"})
    assert appenv.read(str(tmp_path)) == {"OTHER": "x"}, "replaced whole"


@pytest.mark.parametrize("content", ['["K"]', '{"K": 1}', "not json"])
def test_refuses_a_file_that_is_not_an_object_of_strings(tmp_path, content):
    file = tmp_path / ".cache" / "environment.json"
    file.parent.mkdir(parents=True)
    file.write_text(content)
    with pytest.raises(ValueError):
        appenv.read(str(tmp_path))


@pytest.mark.parametrize("planted", [".cache", ".cache/environment.json"])
def test_a_link_planted_on_the_way_is_never_followed(tmp_path, planted):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "environment.json").write_text('{"STOLEN": "yes"}')
    app = tmp_path / "app"
    link = app / planted
    link.parent.mkdir(parents=True, exist_ok=True)
    targets = {".cache": outside, ".cache/environment.json": outside / "environment.json"}
    os.symlink(targets[planted], link)
    with pytest.raises(OSError):
        appenv.read(str(app))
    if planted.endswith(".json"):
        appenv.write(str(app), {"K": "V"})
        assert not link.is_symlink() and appenv.read(str(app)) == {"K": "V"}, "a link in place of the file is replaced, not followed"
    else:
        with pytest.raises(OSError):
            appenv.write(str(app), {"K": "V"})
    assert (outside / "environment.json").read_text() == '{"STOLEN": "yes"}'
