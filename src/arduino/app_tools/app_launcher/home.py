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

logger = logging.getLogger("arduino.app_tools.app_launcher")

DEFAULT_SKELETON = "/etc/skel"
"""The dotfiles the image gives every app: the Streamlit theme and the ALSA wrappers."""


def path(app: str) -> str:
    """The home folder of the app."""
    return os.path.join(app, ".cache", "home")


def provision(app: str, skeleton: str = DEFAULT_SKELETON) -> str:
    """Creates the app home when missing and mirrors the skeleton into it; returns the home.

    Raises:
        OSError: when the home cannot be created.
    """
    home = path(app)
    os.makedirs(home, exist_ok=True)
    for folder, subfolders, files in os.walk(skeleton):
        target = os.path.normpath(os.path.join(home, os.path.relpath(folder, skeleton)))
        for name in [*subfolders, *files]:
            entry = os.path.join(target, name)
            try:
                if name in files:
                    _link(os.path.join(folder, name), entry)
                else:
                    os.makedirs(entry, exist_ok=True)
            except OSError as e:
                # An app may have replaced a skeleton entry with something else: its home, its problem
                logger.warning(f"Cannot mirror the skeleton at {entry}: {e}")
    return home


def _link(source: str, link: str) -> None:
    if os.path.islink(link) and os.readlink(link) == source:
        return
    pending = f"{link}.{os.getpid()}"
    with suppress(FileNotFoundError):
        os.unlink(pending)
    os.symlink(source, pending)
    os.replace(pending, link)
