# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model-state persistence keyed by name + resolved-config fingerprint."""

import os
import pickle
import re
import tempfile
from pathlib import Path

from .events import logger

STATE_DIR_ENV = "ANOMALY_DETECTION_STATE_DIR"
DEFAULT_STATE_DIR = "~/.arduino-bricks/anomaly_detection"


def state_path(name: str) -> Path:
    directory = Path(os.environ.get(STATE_DIR_ENV, DEFAULT_STATE_DIR)).expanduser()
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return directory / f"{safe}.pkl"


def save_state(name: str, fingerprint: str, state: object):
    """Atomically persist state; failures are logged, never raised (detection must go on)."""
    path = state_path(name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps({"fingerprint": fingerprint, "state": state})
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"'{name}': could not persist model state: {e}")


def load_state(name: str, fingerprint: str) -> object | None:
    """Restore persisted state, or None when absent or fitted under a different config."""
    path = state_path(name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as e:
        logger.warning(f"'{name}': could not restore model state: {e}")
        return None
    if payload.get("fingerprint") != fingerprint:
        logger.info(f"'{name}': configuration changed, persisted state invalidated; detector re-warms")
        return None
    return payload.get("state")
