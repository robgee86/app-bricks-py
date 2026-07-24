# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""AnomalyDetection: self-calibrating anomaly detection for a single signal."""

import threading
import time

from arduino.app_utils import brick

from .config import AUTOSAVE_INTERVAL_S
from .events import CallbackRegistry, Stats
from .persistence import load_state, save_state
from .pipeline import SignalPipeline


@brick
class AnomalyDetection:
    """Watches one signal and learns what normal looks like, no thresholds required.

    Push values as they arrive; the AnomalyDetection self-calibrates during warm-up, then fires
    callbacks on anomalies, recoveries, level shifts, predicted limit breaches and
    stalled feeds. Hard `limits` are live from the very first push.

    Examples:
        >>> m = AnomalyDetection("temperature", limits=(0, 90))
        >>> m.on_anomaly(lambda e: print(e.metric, e.value, e.score))
        >>> m.push(23.4)
    """

    def __init__(
        self,
        name: str,
        *,
        sensitivity: str = "medium",
        period: str | None = None,
        limits: tuple | None = None,
        bucket: str = "auto",
        model: str | None = None,
        forecast: str | None = None,
        drift: str = "page_hinkley",
        detector=None,
        persist: bool = True,
    ):
        """Create a monitor for one named signal.

        Args:
            name (str): Signal name, used in events and as the persistence key.
            sensitivity (str): "low", "medium" or "high"; tunes every internal tolerance,
                never which detector runs.
            period (str): Declared seasonality (e.g. "1d") for signals that repeat.
            limits (tuple): Hard (lo, hi) safety bounds, active from the first push;
                one side may be None.
            bucket (str): Seasonal rollup: "auto" (default), "off", or a duration like "30s".
            model (str): "sarimax" for autocorrelated seasonal signals (requires period).
            forecast (str): Horizon (e.g. "30m") to project ahead and fire on_forecast when
                a limit breach is predicted; requires limits.
            drift (str): Level-shift detector, "page_hinkley" (default) or "adwin".
            detector: Bring-your-own river detector replacing the learned one.
            persist (bool): Auto-save/restore model state across restarts.
        """
        if not name or not isinstance(name, str):
            raise ValueError("AnomalyDetection needs a non-empty name")
        self.name = name
        self.persist = persist
        self._lock = threading.RLock()
        self._callbacks = CallbackRegistry()
        self._dirty = False
        self._last_save = time.monotonic()

        self._pipeline = SignalPipeline(
            name,
            sensitivity=sensitivity,
            period=period,
            limits=limits,
            bucket=bucket,
            model=model,
            forecast=forecast,
            drift=drift,
            detector=detector,
        )
        if persist:
            restored = load_state(name, self._pipeline.fingerprint())
            if isinstance(restored, SignalPipeline):
                self._pipeline = restored

        print(f"{name}: {self._pipeline.echo()}", flush=True)

    # ---- data ingestion ----------------------------------------------------------

    def push(self, value: float, at: float | None = None):
        """Feed one observation; fires any resulting callbacks before returning.

        Args:
            value (float): The observed value.
            at (float): Optional arrival timestamp (epoch seconds), defaults to now.
        """
        at = time.time() if at is None else at
        with self._lock:
            events = self._pipeline.process(float(value), at)
            self._dirty = True
        for event in events:
            self._callbacks.dispatch(event)

    # ---- callbacks ---------------------------------------------------------------

    def on_anomaly(self, callback: callable):
        """Register a callback for anomaly events (learned detections and limit breaches)."""
        self._callbacks.register("anomaly", callback)

    def on_normal(self, callback: callable):
        """Register a callback for recovery events after an anomaly episode."""
        self._callbacks.register("normal", callback)

    def on_forecast(self, callback: callable):
        """Register a callback for predicted limit breaches within the forecast horizon."""
        self._callbacks.register("forecast", callback)

    def on_drift(self, callback: callable):
        """Register a callback for level-shift / cadence-change events."""
        self._callbacks.register("drift", callback)

    def on_stalled(self, callback: callable):
        """Register a callback fired when the feed goes silent."""
        self._callbacks.register("stalled", callback)

    def on_ready(self, callback: callable):
        """Register a callback fired once when warm-up completes."""
        self._callbacks.register("ready", callback)

    # ---- introspection -----------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the learned detector has warmed up (limits never wait for this)."""
        with self._lock:
            return self._pipeline.ready

    @property
    def warmup(self) -> int | None:
        """Warm-up override in points; None means the derived formula applies."""
        return self._pipeline.warmup_override

    @warmup.setter
    def warmup(self, points: int):
        with self._lock:
            self._pipeline.warmup_override = int(points)

    @property
    def stats(self) -> Stats:
        """Live resolved view: intent config plus runtime state and the raw detector."""
        with self._lock:
            pipeline = self._pipeline
            return Stats(
                pipeline.intent,
                ready=pipeline.ready,
                points=pipeline.points,
                bucket_size=pipeline.bucket_size_s,
                detector=pipeline.detector,
            )

    # ---- lifecycle ---------------------------------------------------------------

    def loop(self):
        """Framework loop: staleness watchdog and periodic auto-save."""
        time.sleep(0.5)
        now = time.time()
        with self._lock:
            events = self._pipeline.check_staleness(now)
        for event in events:
            self._callbacks.dispatch(event)
        self._autosave()

    def stop(self):
        """Persist state on clean shutdown."""
        self._save_if_dirty()

    def _autosave(self):
        if time.monotonic() - self._last_save >= AUTOSAVE_INTERVAL_S:
            self._save_if_dirty()

    def _save_if_dirty(self):
        if not self.persist:
            return
        with self._lock:
            if not self._dirty:
                return
            snapshot = self._pipeline
            fingerprint = snapshot.fingerprint()
            self._dirty = False
            save_state(self.name, fingerprint, snapshot)
        self._last_save = time.monotonic()
