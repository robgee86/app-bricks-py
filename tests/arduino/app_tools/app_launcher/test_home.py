# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

from arduino.app_tools.app_launcher import home


def skeleton_of(tmp_path):
    skeleton = tmp_path / "skel"
    (skeleton / ".streamlit").mkdir(parents=True)
    (skeleton / ".streamlit" / "config.toml").write_text("[theme]\n")
    (skeleton / ".asoundrc").write_text("pcm\n")
    return skeleton


def test_the_home_is_under_the_app_cache_and_mirrors_the_skeleton(tmp_path):
    skeleton = skeleton_of(tmp_path)
    app = tmp_path / "app"
    app.mkdir()

    provisioned = home.provision(str(app), str(skeleton))

    assert provisioned == str(app / ".cache" / "home") == home.path(str(app))
    assert (app / ".cache" / "home" / ".streamlit").is_dir() and not (app / ".cache" / "home" / ".streamlit").is_symlink()
    assert os.readlink(app / ".cache" / "home" / ".streamlit" / "config.toml") == str(skeleton / ".streamlit" / "config.toml")
    assert (app / ".cache" / "home" / ".asoundrc").read_text() == "pcm\n"


def test_what_the_app_adds_stays_and_the_skeleton_files_are_the_image_ones_again(tmp_path):
    skeleton = skeleton_of(tmp_path)
    app = tmp_path / "app"
    app.mkdir()
    home.provision(str(app), str(skeleton))
    mine = app / ".cache" / "home" / ".config" / "mine.toml"
    mine.parent.mkdir()
    mine.write_text("kept\n")
    theme = app / ".cache" / "home" / ".streamlit" / "config.toml"
    theme.unlink()
    theme.write_text("[theme]\nbase = 'dark'\n")

    home.provision(str(app), str(skeleton))

    assert mine.read_text() == "kept\n"
    assert theme.is_symlink() and theme.read_text() == "[theme]\n"


def test_a_missing_skeleton_gives_an_empty_home(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    assert os.listdir(home.provision(str(app), str(tmp_path / "nowhere"))) == []
