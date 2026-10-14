# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os

MAX_BYTES = 5 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
"""How much output is read at once."""


class AppLog:
    """Appends an app's raw output to a file, rotated by size into a single previous file, like Docker's json-file driver."""

    def __init__(self, path: str, max_bytes: int = MAX_BYTES, fresh: bool = False) -> None:
        """Opens the log, creating its folder.

        Args:
            path (str): The log file, `app.log`; the rotated one is `app.log.1`.
            max_bytes (int): The size past which the log is rotated before the next write.
            fresh (bool): Start over: the previous content and the rotated file go, so the log holds one run.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._max_bytes = max_bytes
        if fresh:
            try:
                os.remove(f"{path}.1")
            except FileNotFoundError:
                pass
        self._file = open(path, "wb" if fresh else "ab", buffering=0)  # noqa: SIM115
        self._size = self._file.tell()

    def write(self, data: bytes) -> None:
        """Appends data, rotating first when it would exceed the size limit."""
        if self._size and self._size + len(data) > self._max_bytes:
            self._file.close()
            os.replace(self.path, f"{self.path}.1")
            self._file = open(self.path, "wb", buffering=0)  # noqa: SIM115
            self._size = 0
        self._file.write(data)
        self._size += len(data)

    def close(self) -> None:
        """Closes the file."""
        self._file.close()
