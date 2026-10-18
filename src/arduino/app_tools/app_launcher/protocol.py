# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Control socket protocol: one JSON object per line, one request and one response per connection."""

import json
import os
from dataclasses import dataclass, field
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
    env: dict[str, str] = field(default_factory=dict[str, str])
    mode: Mode = Mode.REPLACE


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
        case {"cmd": "start"}:
            return Start(
                app=_app(message.get("app")),
                env=_env(message.get("env", {})),
                mode=_mode(message.get("mode", Mode.REPLACE)),
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


def _app(value: object) -> str:
    match value:
        case str() if os.path.isabs(value):
            return os.path.normpath(value)
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "app must be the absolute path of the app folder")


def _env(value: object) -> dict[str, str]:
    match value:
        case dict():
            env: dict[str, str] = {}
            for item in value.items():
                match item:
                    case (str() as key, str() as text):
                        env[key] = text
                    case _:
                        raise LauncherError(ErrorCode.BAD_REQUEST, "env values must be strings")
            return env
        case _:
            raise LauncherError(ErrorCode.BAD_REQUEST, "env must be an object of strings")


def _mode(value: object) -> Mode:
    try:
        return Mode(value)
    except ValueError:
        raise LauncherError(ErrorCode.BAD_REQUEST, f"mode must be one of {[str(m) for m in Mode]}") from None
