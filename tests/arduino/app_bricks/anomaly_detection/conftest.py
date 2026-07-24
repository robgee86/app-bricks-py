# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import random

import pytest

T0 = 1_000_000.0


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    """Keep persisted model state out of the user's home directory."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("ANOMALY_DETECTION_STATE_DIR", str(state_dir))
    return state_dir


@pytest.fixture(autouse=True)
def seeded_random():
    random.seed(42)


class Recorder:
    """Collects events per kind for assertions."""

    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

    def attach(self, target):
        """Register this recorder on every callback the brick exposes."""
        for kind in ("anomaly", "normal", "forecast", "drift", "stalled", "ready"):
            register = getattr(target, f"on_{kind}", None)
            if register is not None:
                register(self)
        return self

    def kinds(self):
        return [event.kind for event in self.events]

    def of_kind(self, kind):
        return [event for event in self.events if event.kind == kind]


@pytest.fixture
def recorder():
    return Recorder()
