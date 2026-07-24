# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Event record, callback dispatch and the occurrence-gating / hysteresis state machine."""

from collections import deque
from dataclasses import dataclass, field

from arduino.app_utils import Logger

logger = Logger("AnomalyDetection")

KINDS = ("anomaly", "normal", "forecast", "drift", "stalled", "ready")


class Stats(dict):
    """Dict with attribute access; missing entries read as None (fields are 'when applicable')."""

    def __getattr__(self, name):
        return self.get(name)


@dataclass(frozen=True)
class AnomalyEvent:
    """Flat event record, identical shape from AnomalyDetection and AnomalyDetectionGroup callbacks.

    Attributes:
        metric (str): AnomalyDetection/AnomalyDetectionGroup name ("rig", or "rig.vibration" for per-member events).
        kind (str): One of "anomaly", "normal", "forecast", "drift", "stalled", "ready".
        value: Raw pushed value (dict of members for AnomalyDetectionGroup joint events).
        expected: What the detector considered normal at this point; None when undefined.
        score (float): 0-1 normalized anomaly score.
        at (float): Arrival timestamp (epoch seconds) of the triggering push.
        stats (Stats): Raw indicators (z, residual, threshold, reason, detector, ...).
        summary (str | None): Beginner-facing one-liner, set on AnomalyDetectionGroup joint events.
    """

    metric: str
    kind: str
    value: object
    expected: object
    score: float
    at: float
    stats: Stats = field(default_factory=Stats)
    summary: str | None = None


class CallbackRegistry:
    """Per-kind callback lists; dispatch never lets a user callback break the pipeline."""

    def __init__(self):
        self._callbacks = {kind: [] for kind in KINDS}

    def register(self, kind: str, callback: callable):
        self._callbacks[kind].append(callback)

    def dispatch(self, event: AnomalyEvent):
        for callback in self._callbacks[event.kind]:
            try:
                callback(event)
            except Exception:
                logger.exception(f"Callback for '{event.metric}' ({event.kind}) raised")


class EpisodeGate:
    """Occurrence gating, emitted-score floor and recovery hysteresis for one event stream.

    Fires "anomaly" when X of the last Y evaluations were anomalous (and the score clears
    the floor), then stays silent until N consecutive normal evaluations fire "normal".
    """

    def __init__(self, gate: tuple[int, int], hysteresis: int, score_floor: float):
        self.gate = gate
        self.hysteresis = hysteresis
        self.score_floor = score_floor
        self._recent = deque(maxlen=gate[1])
        self._in_anomaly = False
        self._normal_streak = 0

    @property
    def in_anomaly(self) -> bool:
        return self._in_anomaly

    def observe(self, anomalous: bool, score: float, gate: tuple[int, int] | None = None) -> str | None:
        """Record one evaluation; returns "anomaly", "normal" or None.

        A per-call gate override supports lone-signal damping without altering the configured gate.
        """
        needed, window = gate or self.gate
        if self._recent.maxlen != window:
            self._recent = deque(self._recent, maxlen=window)
        self._recent.append(bool(anomalous))

        if not self._in_anomaly:
            if anomalous and sum(self._recent) >= needed and score >= self.score_floor:
                self._in_anomaly = True
                self._normal_streak = 0
                return "anomaly"
            return None

        if anomalous:
            self._normal_streak = 0
            return None
        self._normal_streak += 1
        if self._normal_streak >= self.hysteresis:
            self._in_anomaly = False
            self._normal_streak = 0
            self._recent.clear()
            return "normal"
        return None

    def force_anomalous(self):
        """Enter the anomaly episode directly (hard-limit breaches bypass gating)."""
        self._in_anomaly = True
        self._normal_streak = 0
