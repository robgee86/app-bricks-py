# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Where the running app's folder is. Stdlib only, cheap enough for import time."""

import os
import sys

MANIFEST = "app.yaml"
"""The file at the root of every app."""


def app_folder(*parts: str) -> str:
    """Returns the folder of the running app, or a path inside it.

    The app folder is the nearest folder holding an app.yaml above the running script, else above the working directory.
    Without one, it is the working directory itself: the app folder under the launcher, wherever a developer runs the app from.

    Args:
        *parts (str): The path components below the app folder, none for the folder itself.

    Returns:
        str: The absolute path.
    """
    return os.path.join(_locate(), *parts)


def _locate() -> str:
    cwd = os.getcwd()
    for start in (_main_script_folder(), cwd):
        if start is not None and (found := _nearest_app(start)) is not None:
            return found
    return cwd


def _main_script_folder() -> str | None:
    file = getattr(sys.modules.get("__main__"), "__file__", None)
    return os.path.dirname(os.path.abspath(file)) if file else None


def _nearest_app(start: str) -> str | None:
    folder = os.path.abspath(start)
    while not os.path.isfile(os.path.join(folder, MANIFEST)):
        parent = os.path.dirname(folder)
        if parent == folder:
            return None
        folder = parent
    return folder
