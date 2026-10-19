# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The home of an app: a folder of its own under its .cache, mirroring the image skeleton at every start. Stdlib only.

One container runs every app, so a shared home would carry what one app writes, a uv config or a Python package, into the
next app and its preparation. Each app gets its own instead, as a new user would: the skeleton folders are created in it
and the skeleton files linked, so an image upgrade reaches every app, while what the app adds stays.
"""

import logging
import os
from contextlib import suppress

from . import folders

logger = logging.getLogger("arduino.app_tools.app_launcher")

DEFAULT_SKELETON = "/etc/skel"
"""The dotfiles the image gives every app: the Streamlit theme and the ALSA wrappers."""


def path(app: str) -> str:
    """The home folder of the app."""
    return os.path.join(app, ".cache", "home")


def provision(app: str, skeleton: str = DEFAULT_SKELETON) -> str:
    """Creates the app home when missing and mirrors the skeleton into it; returns the home.

    Raises:
        OSError: when the home cannot be created, a link planted in its place included.
    """
    home = path(app)
    home_fd = folders.open_folder(app, ".cache", "home")
    try:
        with suppress(FileNotFoundError):
            _mirror(skeleton, home_fd)
    finally:
        os.close(home_fd)
    return home


def _mirror(source: str, target_fd: int) -> None:
    for entry in os.scandir(source):
        try:
            if entry.is_dir(follow_symlinks=False):
                fd = folders.descend(target_fd, entry.name)
                try:
                    _mirror(entry.path, fd)
                finally:
                    os.close(fd)
            else:
                _link(entry.path, entry.name, target_fd)
        except OSError as e:
            # An app may have replaced a skeleton entry with something else: its home, its problem
            logger.warning(f"Cannot mirror the skeleton entry {entry.name}: {e}")


def _link(source: str, name: str, folder_fd: int) -> None:
    with suppress(OSError):
        if os.readlink(name, dir_fd=folder_fd) == source:
            return
    pending = f"{name}.{os.getpid()}"
    with suppress(FileNotFoundError):
        os.unlink(pending, dir_fd=folder_fd)
    os.symlink(source, pending, dir_fd=folder_fd)
    os.replace(pending, name, src_dir_fd=folder_fd, dst_dir_fd=folder_fd)
