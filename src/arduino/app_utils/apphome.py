# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Where the running app's folder is. Stdlib only, cheap enough for import time."""

import os


def app_folder(*parts: str) -> str:
    """Returns the folder of the running app, or a path inside it.

    Args:
        *parts (str): The path components below the app folder, none for the folder itself.

    Returns:
        str: The path under APP_HOME when it names an existing folder, else under the current working directory, the app folder
            under the launcher and wherever a developer runs the app from.
    """
    home = os.environ.get("APP_HOME")
    return os.path.join(home if home and os.path.isdir(home) else os.getcwd(), *parts)
