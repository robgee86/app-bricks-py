# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Control socket protocol: one JSON object per line, one request and one response per connection."""

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_SOCKET = "/run/arduino-app-cli/app-launcher.sock"
MAX_LINE_BYTES = 1024 * 1024


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"
    FORBIDDEN = "forbidden"
    BUSY = "busy"
    PREPARE_FAILED = "prepare_failed"
    INTERNAL = "internal"


class LauncherError(Exception):
    """A request the launcher refuses or fails, reported to the client with its code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Mode(StrEnum):
    REPLACE = "replace"
    ADD = "add"


@dataclass(frozen=True)
class Start:
    app: str
    mode: Mode = Mode.REPLACE
    ports: tuple[int, ...] = ()
    """The TCP ports the app may bind, the ones it declares."""


@dataclass(frozen=True)
class Stop:
    app: str | None = None


@dataclass(frozen=True)
class Status:
    pass


@dataclass(frozen=True)
class Ping:
    pass


Request = Start | Stop | Status | Ping


def encode(message: dict[str, Any]) -> bytes:
    """Serializes a message as one line."""
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def ok(**fields: Any) -> dict[str, Any]:
    """Builds a success response."""
    return {"ok": True, **fields}


def error(err: LauncherError) -> dict[str, Any]:
    """Builds an error response."""
    return {"ok": False, "error": {"code": str(err.code), "message": err.message}}


def decode_request(line: bytes) -> Request:
    """Parses and validates one request line.

    Raises:
        LauncherError: bad_request when the line is not a valid request.
    """
    try:
        message: Any = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LauncherError(ErrorCode.BAD_REQUEST, f"invalid JSON: {e}") from None
    match message:
        case {"cmd": "start", "env": _}:
            # Never through the launcher process: what it read of one app would linger in the memory the next is forked from
            raise LauncherError(ErrorCode.BAD_REQUEST, "env is not taken, the app environment is read from <app>/.cache/launcher/environment.json")
        case {"cmd": "start"}:
            return Start(
                app=_app(message.get("app")),
                mode=_mode(message.get("mode", Mode.REPLACE)),
                ports=_ports(message.get("ports", [])),
            )
        case {"cmd": "stop"}:
            app = message.get("app")
            return Stop(app=None if app is None else _app(app))
        case {"cmd": "status"}:
            return Status()
        case {"cmd": "ping"}:
            return Ping()
        case {"cmd": cmd}:
            raise LauncherError(ErrorCode.BAD_REQUEST, f"unknown cmd: {cmd!r}")
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "a request must be a JSON object with a cmd")


def _ports(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= 65535 for p in value):
        raise LauncherError(ErrorCode.BAD_REQUEST, "ports must be a list of port numbers")
    return tuple(sorted(set(value)))


def _app(value: object) -> str:
    match value:
        case str() if os.path.isabs(value):
            return os.path.normpath(value)
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "app must be the absolute path of the app folder")


def _mode(value: object) -> Mode:
    try:
        return Mode(value)
    except ValueError:
        raise LauncherError(ErrorCode.BAD_REQUEST, f"mode must be one of {[str(m) for m in Mode]}") from None
