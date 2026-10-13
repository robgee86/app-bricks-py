# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Where the running app's folder is. Stdlib only, cheap enough for import time."""

import os

DEFAULT_APP_ROOT = "/app"


def app_root() -> str:
    """Returns the folder of the running app.

    Returns:
        str: APP_HOME when it names an existing folder, else "/app", where the classic container mounts the app.
    """
    home = os.environ.get("APP_HOME")
    return home if home and os.path.isdir(home) else DEFAULT_APP_ROOT


def app_path(*parts: str) -> str:
    """Returns a path inside the folder of the running app.

    Args:
        *parts (str): The path components below the app folder.

    Returns:
        str: The joined path.
    """
    return os.path.join(app_root(), *parts)
