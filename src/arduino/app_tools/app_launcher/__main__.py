# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import sys

from arduino.app_tools.app_launcher import client, prepare, supervisor


def main() -> int:
    """Entry point of `arduino-app-launcher`: runs the launcher without arguments, talks to it with a command; `record-preparation`
    is for `run.sh prepare`, which ends by recording what it prepared."""
    match sys.argv[1:]:
        case []:
            return supervisor.main()
        case ["record-preparation", *rest]:
            return prepare.main(["record", *rest])
        case argv:
            return client.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
