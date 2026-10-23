# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Command line client of the control socket: `arduino-app-launcher ping|status|start|stop`. Stdlib only."""

import argparse
import json
import os
import socket
import sys
from collections.abc import Sequence
from typing import Any

from .protocol import DEFAULT_SOCKET


def request(message: dict[str, Any], socket_path: str = DEFAULT_SOCKET, timeout_s: float = 60.0) -> dict[str, Any]:
    """Sends one request and returns the response.

    Raises:
        OSError: when the launcher cannot be reached or does not answer in time.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        sock.sendall(json.dumps(message).encode() + b"\n")
        return json.loads(sock.makefile("rb").readline())


def main(argv: Sequence[str]) -> int:
    """Parses a command, sends it and prints the JSON response; returns 0 when the launcher accepted it."""
    parser = argparse.ArgumentParser(prog="arduino-app-launcher", description="Runs the launcher when called without a command.")
    parser.add_argument("--socket", default=os.environ.get("APP_LAUNCHER_SOCKET", DEFAULT_SOCKET), help="control socket path")
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("ping", help="check that the launcher answers")
    commands.add_parser("status", help="show the running app and the preload")
    start = commands.add_parser("start", help="start the app, stopping the running one first")
    start.add_argument("app", help="app folder")
    start.add_argument("--env", action="append", default=[], metavar="K=V", help="app environment variable, repeatable")
    commands.add_parser("stop", help="stop the running app")
    args = parser.parse_args(argv)

    message: dict[str, Any] = {"cmd": args.cmd}
    if args.cmd == "start":
        if not all("=" in item for item in args.env):
            parser.error("--env takes K=V")
        env = {key: value for key, _, value in (item.partition("=") for item in args.env)}
        message |= {"app": os.path.abspath(args.app), "env": env}
    try:
        response = request(message, args.socket)
    except OSError as e:
        print(f"Cannot reach the launcher at {args.socket}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(response, indent=2))
    return 0 if response.get("ok") else 1
