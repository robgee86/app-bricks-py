# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os
from typing import BinaryIO

from . import folders

MAX_BYTES = 5 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
"""How much output is read at once."""
NAME = "app.log"


def path(app: str) -> str:
    """The log file of the app, `app.log`; the rotated one is `app.log.1`."""
    return os.path.join(app, ".cache", "logs", NAME)


class AppLog:
    """Appends an app's raw output to a file, rotated by size into a single previous file, like Docker's json-file driver."""

    def __init__(self, app: str, max_bytes: int = MAX_BYTES, fresh: bool = False) -> None:
        """Opens the log of the app, creating its folder; see `path()`.

        Args:
            app (str): The app folder.
            max_bytes (int): The size past which the log is rotated before the next write.
            fresh (bool): Start over: the previous content and the rotated file go, so the log holds one run.

        Raises:
            OSError: when the log cannot be opened, a link planted in its place included.
        """
        self.path = path(app)
        self._max_bytes = max_bytes
        self._folder = folders.open_folder(app, ".cache", "logs")
        try:
            if fresh:
                try:
                    os.unlink(f"{NAME}.1", dir_fd=self._folder)
                except FileNotFoundError:
                    pass
            self._file = self._open(truncate=fresh)
        except BaseException:
            os.close(self._folder)
            raise
        self._size = os.fstat(self._file.fileno()).st_size

    def _open(self, truncate: bool) -> BinaryIO:
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if truncate else os.O_APPEND)
        return os.fdopen(folders.open_file(self._folder, NAME, flags), "wb", buffering=0)

    def write(self, data: bytes | memoryview) -> None:
        """Appends data, rotating first when it would exceed the size limit."""
        if self._size and self._size + len(data) > self._max_bytes:
            self._file.close()
            os.replace(NAME, f"{NAME}.1", src_dir_fd=self._folder, dst_dir_fd=self._folder)
            self._file = self._open(truncate=True)
            self._size = 0
        self._file.write(data)
        self._size += len(data)

    def close(self) -> None:
        """Closes the file."""
        self._file.close()
        os.close(self._folder)
