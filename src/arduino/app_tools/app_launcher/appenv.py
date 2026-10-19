# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The environment an app runs with: a file the daemon writes in the app folder, read by the app process once confined. Stdlib only.

The launcher process never reads it: what it held of one app, its secrets included, would stay in the memory the next app
is forked from. Every step below the app folder is opened refusing links, see `folders`.
"""

import json
import os
from collections.abc import Mapping

from . import folders

NAME = "environment.json"


def path(app: str) -> str:
    """The environment file of the app, a JSON object of strings."""
    return os.path.join(app, ".cache", "launcher", NAME)


def read(app: str) -> dict[str, str]:
    """Reads the environment of the app, empty when none was written.

    Raises:
        OSError: when the file cannot be opened, a link in its place included.
        ValueError: when the file does not hold an object of strings.
    """
    folder_fd = folders.open_folder(app, ".cache", "launcher")
    try:
        try:
            fd = folders.open_file(folder_fd, NAME, os.O_RDONLY)
        except FileNotFoundError:
            return {}
    finally:
        os.close(folder_fd)
    with os.fdopen(fd, "rb") as file:
        return validate(json.load(file))


def write(app: str, env: Mapping[str, str]) -> None:
    """Writes the environment of the app, readable by its uid only, replacing the previous one whole.

    Raises:
        OSError: when the file cannot be written, a link on the way included.
    """
    folder_fd = folders.open_folder(app, ".cache", "launcher")
    try:
        pending = f"{NAME}.{os.getpid()}"
        fd = folders.open_file(folder_fd, pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(dict(env), file)
        os.replace(pending, NAME, src_dir_fd=folder_fd, dst_dir_fd=folder_fd)
    finally:
        os.close(folder_fd)


def validate(value: object) -> dict[str, str]:
    """Returns the value as an environment.

    Raises:
        ValueError: when it is not an object of strings.
    """
    match value:
        case dict():
            env: dict[str, str] = {}
            for item in value.items():
                match item:
                    case (str() as key, str() as text):
                        env[key] = text
                    case _:
                        raise ValueError("the environment values must be strings")
            return env
        case _:
            raise ValueError("the environment must be an object of strings")
