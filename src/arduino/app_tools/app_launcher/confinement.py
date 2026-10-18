# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Confines an app process to its own folder and the runtime with Landlock, the kernel's unprivileged sandbox. Stdlib only.

The process restricts itself, right before any app code runs: no capability, no profile loaded by root, nothing to undo. It
is an allow-list: the app folder is read-write, the runtime folders are read-only, devices and the shared folders are
read-write, everything else, the other apps above all, is out of reach. A later ruleset can only tighten it, and it follows
the process through exec and fork. What Landlock does not mediate: connecting to a Unix socket by its path, and on kernels
before ABI 6 signals and abstract sockets.

As a command, it confines itself and becomes another program, for the preparation: `python -m ...confinement --app A -- cmd`.
"""

import argparse
import ctypes
import errno
import os
import stat
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446
_PR_SET_NO_NEW_PRIVS = 38
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH, _RULE_NET_PORT = 1, 2

# Filesystem access rights, with the ABI that introduced them
FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FS_ABI1 = (1 << 13) - 1
_FS_REFER, _FS_TRUNCATE, _FS_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_NET_BIND_TCP = 1 << 0
_SCOPE_ABSTRACT_UNIX_SOCKET, _SCOPE_SIGNAL = 1 << 0, 1 << 1

READ_ONLY = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR

RUNTIME_READ_ONLY = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/sys", "/proc", "/run.sh")
"""The image and the kernel views an app reads; run.sh is what the preparation runs."""
RUNTIME_READ_WRITE = ("/dev", "/tmp")
"""Devices, with their ioctls, and the temporary folder. /dev/shm holds the semaphores of multiprocessing."""


class Unavailable(OSError):
    """The kernel has no Landlock, or it is disabled."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _NetPort(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


@dataclass(frozen=True)
class Policy:
    """What a confined process may reach: its app folder in full, the rest as listed."""

    app: str
    ports: tuple[int, ...] = ()
    """The TCP ports it may bind besides an ephemeral one, none when it serves nothing."""
    read_only: tuple[str, ...] = RUNTIME_READ_ONLY
    read_write: tuple[str, ...] = RUNTIME_READ_WRITE

    @classmethod
    def for_app(cls, app: str, ports: Iterable[int], env: Mapping[str, str]) -> "Policy":
        """The policy of an app: the runtime, plus the models folder to read and the home and runtime folders to write."""
        read_only = RUNTIME_READ_ONLY + tuple(filter(None, [env.get("MODELS_PATH")]))
        read_write = RUNTIME_READ_WRITE + tuple(filter(None, [env.get("HOME"), env.get("XDG_RUNTIME_DIR")]))
        return cls(app=os.path.abspath(app), ports=tuple(ports), read_only=read_only, read_write=read_write)


def abi() -> int:
    """The Landlock ABI version of the running kernel.

    Raises:
        Unavailable: when the kernel has no Landlock.
    """
    if sys.platform != "linux":
        raise Unavailable(errno.ENOSYS, "Landlock is a Linux kernel feature")
    try:
        return _syscall(_SYS_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    except OSError as e:
        if e.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
            raise Unavailable(e.errno, "the kernel has no Landlock") from None
        raise


def apply(policy: Policy) -> int:
    """Confines the calling thread, and every thread and process it creates from now on, to the policy; returns the ABI used.

    Raises:
        Unavailable: when the kernel has no Landlock.
        OSError: when the kernel refuses the ruleset.
    """
    version = abi()
    fs = _FS_ABI1 | (_FS_REFER if version >= 2 else 0) | (_FS_TRUNCATE if version >= 3 else 0) | (_FS_IOCTL_DEV if version >= 5 else 0)
    net = _NET_BIND_TCP if version >= 4 else 0
    scoped = (_SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL) if version >= 6 else 0
    attr = _RulesetAttr(fs, net, scoped)
    # Older kernels know a shorter attr: hand them only the fields they understand
    size = ctypes.sizeof(attr) if version >= 6 else 16 if version >= 4 else 8
    ruleset = _syscall(_SYS_CREATE_RULESET, ctypes.byref(attr), size, 0)
    try:
        for path, access in [(policy.app, fs), *((p, fs) for p in policy.read_write), *((p, READ_ONLY) for p in policy.read_only)]:
            _allow_path(ruleset, path, access & fs)
        # Port 0, an ephemeral port of the kernel's choice: how libraries open a helper server of their own, unreachable from outside
        for port in (0, *policy.ports) if net else ():
            rule = _NetPort(_NET_BIND_TCP, port)
            _syscall(_SYS_ADD_RULE, ruleset, _RULE_NET_PORT, ctypes.byref(rule), 0)
        if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
        _syscall(_SYS_RESTRICT_SELF, ruleset, 0)
    finally:
        os.close(ruleset)
    return version


_FILE_ACCESS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV
"""The rights that apply to a file: the kernel refuses a rule granting a file the rights of a folder."""


def _allow_path(ruleset: int, path: str, access: int) -> None:
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except FileNotFoundError:
        return
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            access &= _FILE_ACCESS
        rule = _PathBeneath(access, fd)
        _syscall(_SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), 0)
    finally:
        os.close(fd)


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _syscall(number: int, *args: object) -> int:
    result = _libc().syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def main(argv: list[str]) -> int:
    """Confines this process to an app, then becomes the command: how the preparation of an app runs."""
    parser = argparse.ArgumentParser(prog="python -m arduino.app_tools.app_launcher.confinement", description=main.__doc__)
    parser.add_argument("--app", required=True, help="the app folder")
    parser.add_argument("--port", action="append", type=int, default=[], help="a TCP port the command may bind, repeatable")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    apply(Policy.for_app(args.app, args.port, os.environ))
    os.execvp(command[0], command)
    return 1  # pragma: no cover, execvp does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
