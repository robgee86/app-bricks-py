# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Folders the launcher keeps under an app folder, which the confined app can rewrite, links included. Stdlib only.

The supervisor is not confined, so a link an app plants where the launcher writes would carry the write anywhere the
supervisor reaches. Every step below the app folder is opened refusing links, and files are reached through the folder
descriptor, never through a path the app could redirect.
"""

import os
from contextlib import suppress

FOLDER = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def open_folder(app: str, *parts: str) -> int:
    """Opens the folder `parts` deep under the app folder, creating what is missing; returns its descriptor.

    Raises:
        OSError: when a link stands where a folder is expected, or a folder cannot be created or opened.
    """
    fd = os.open(app, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in parts:
            inner = descend(fd, name)
            os.close(fd)
            fd = inner
    except BaseException:
        os.close(fd)
        raise
    return fd


def descend(folder_fd: int, name: str) -> int:
    """Opens the folder `name` inside the open folder, creating it when missing; returns its descriptor.

    Raises:
        OSError: when a link stands in its place, or it cannot be created or opened.
    """
    with suppress(FileExistsError):
        os.mkdir(name, dir_fd=folder_fd)
    return os.open(name, FOLDER, dir_fd=folder_fd)


def open_file(folder_fd: int, name: str, flags: int, mode: int = 0o644) -> int:
    """Opens the file `name` inside the open folder with the flags, refusing a link in its place; returns its descriptor.

    Raises:
        OSError: when a link stands in its place, or it cannot be opened.
    """
    return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=folder_fd)
