# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""AnomalyDetectionGroup: joint and relationship anomaly detection over 2-5 related signals."""

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import astuple

from river import anomaly, drift, linear_model, optim, stats as river_stats

from arduino.app_utils import brick

from .config import (
    AUTOSAVE_INTERVAL_S,
    CADENCE_RESOLUTION_POINTS,
    GROUP_JOINT_WARMUP,
    GROUP_RELATIONSHIP_WARMUP,
    MAX_GROUP_MEMBERS,
    broadcast_member_kwarg,
    config_fingerprint,
    parse_duration,
    raise_gate,
    reject_flattened_limits,
    reject_group_kwarg_array,
    resolve_sensitivity,
)
from .detectors import CUTOFF_WINDOW, quantile_filter
from .events import AnomalyEvent, CallbackRegistry, EpisodeGate, Stats
from .persistence import load_state, save_state
from .pipeline import SignalPipeline

# Internal recovery thresholds for relationship episodes (deviation below these counts as normal).
_PAIR_RECOVERY_DEVIATION = 0.25
_MULTI_RECOVERY_Z = 2.0


class _ScaleWindow:
    """Running min-max scaling over a rolling P1-P99 window of raw values."""

    def __init__(self, size: int = 500):
        self._values = deque(maxlen=size)

    def add(self, value: float):
        self._values.append(value)

    def bounds(self) -> tuple[float, float]:
        ordered = sorted(self._values)
        if not ordered:
            return 0.0, 1.0
        p1 = ordered[max(0, int(0.01 * (len(ordered) - 1)))]
        p99 = ordered[int(0.99 * (len(ordered) - 1))]
        return p1, p99

    def scale(self, value: float) -> float:
        p1, p99 = self.bounds()
        if p99 <= p1:
            return 0.5
        return min(1.0, max(0.0, (value - p1) / (p99 - p1)))

    def unscale(self, scaled: float) -> float:
        p1, p99 = self.bounds()
        return p1 + scaled * (p99 - p1) if p99 > p1 else p1


class _JointWatcher:
    """Is this combination of values abnormal? HalfSpaceTrees over the scaled vector."""

    def __init__(self, signals: list[str], quantile: float):
        hst = anomaly.HalfSpaceTrees(n_trees=10, height=8, window_size=250, limits={s: (0.0, 1.0) for s in signals}, seed=42)
        self._filter = quantile_filter(hst, quantile)
        self._scores = deque(maxlen=250)

    def set_quantile(self, quantile: float):
        self._filter.quantile = river_stats.RollingQuantile(q=quantile, window_size=CUTOFF_WINDOW)

    def evaluate(self, scaled: dict) -> tuple[float, bool]:
        """Returns (score min-max normalized over the recent window, anomalous)."""
        raw = self._filter.score_one(scaled)
        anomalous = self._filter.classify(raw)
        self._filter.learn_one(scaled)
        self._scores.append(raw)
        low, high = min(self._scores), max(self._scores)
        norm = (raw - low) / (high - low) if high > low else 0.0
        return norm, anomalous

    @property
    def detector(self):
        return self._filter.anomaly_detector


class _EwmaMoments:
    """Exponentially weighted means, variances and covariance of a value pair."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mx = self.my = 0.0
        self.vxx = self.vyy = self.vxy = 0.0
        self.n = 0

    def update(self, x: float, y: float):
        self.n += 1
        if self.n == 1:
            self.mx, self.my = x, y
            return
        a = self.alpha
        dx, dy = x - self.mx, y - self.my
        self.mx += a * dx
        self.my += a * dy
        self.vxx = (1 - a) * (self.vxx + a * dx * dx)
        self.vyy = (1 - a) * (self.vyy + a * dy * dy)
        self.vxy = (1 - a) * (self.vxy + a * dx * dy)

    @property
    def r(self) -> float:
        denominator = math.sqrt(self.vxx * self.vyy)
        return self.vxy / denominator if denominator > 0 else 0.0


class _PairRelationship:
    """Has the usual coupling broken? Incremental Pearson + PageHinkley on the r stream."""

    def __init__(self, ph_threshold: float):
        self.fast = _EwmaMoments(alpha=0.05)
        self.slow = _EwmaMoments(alpha=0.005)
        self._ph = drift.PageHinkley(delta=0.005, threshold=ph_threshold, min_instances=60)

    def set_threshold(self, ph_threshold: float):
        self._ph = drift.PageHinkley(delta=0.005, threshold=ph_threshold, min_instances=60)

    def evaluate(self, x: float, y: float) -> dict:
        self.fast.update(x, y)
        self.slow.update(x, y)
        self._ph.update(self.fast.r)
        deviation = abs(self.fast.r - self.slow.r)
        return {
            "fired": self._ph.drift_detected,
            "score": min(1.0, deviation / 0.5),
            "deviating": None,
            "expected": self.slow.r,
            "recovered": deviation < _PAIR_RECOVERY_DEVIATION,
            "stats": {"r": round(self.fast.r, 4), "expected_r": round(self.slow.r, 4)},
        }

    @property
    def detector(self):
        return self._ph


class _MultiRelationship:
    """Online ridge regression of each member on the others; PageHinkley per residual stream."""

    def __init__(self, signals: list[str], ph_threshold: float):
        self.signals = signals
        # Light ridge + a faster learning rate: inputs are min-max scaled, and the model
        # must converge within the 120-evaluation relationship warm-up.
        self._models = {s: linear_model.LinearRegression(optimizer=optim.SGD(0.1), l2=0.01, intercept_lr=0.1) for s in signals}
        self._ph = {s: drift.PageHinkley(delta=0.005, threshold=ph_threshold, min_instances=60) for s in signals}
        # Moments of |residual|: a decoupled member shifts the magnitude's mean even when
        # the raw residual stays zero-mean. The alpha balances converging within the
        # relationship warm-up against adapting so fast that a break stops looking unusual.
        self._residual_stats = {s: _EwmaMoments(alpha=0.02) for s in signals}

    def set_threshold(self, ph_threshold: float):
        self._ph = {s: drift.PageHinkley(delta=0.005, threshold=ph_threshold, min_instances=60) for s in self.signals}

    def evaluate(self, scaled: dict) -> dict:
        fired, z_by_member, pred_by_member = [], {}, {}
        for signal in self.signals:
            features = {other: scaled[other] for other in self.signals if other != signal}
            predicted = self._models[signal].predict_one(features)
            magnitude = abs(scaled[signal] - predicted)
            moments = self._residual_stats[signal]
            sigma = math.sqrt(moments.vxx)
            z = (magnitude - moments.mx) / sigma if sigma > 1e-9 else 0.0
            z_by_member[signal] = z
            moments.update(magnitude, magnitude)
            pred_by_member[signal] = predicted
            # PageHinkley on the normalized residual stream: z is scale-free, so T_sens
            # means the same thing regardless of the signals' units.
            self._ph[signal].update(z)
            if self._ph[signal].drift_detected:
                fired.append(signal)
            self._models[signal].learn_one(features, scaled[signal])
        deviating = max(fired or z_by_member, key=lambda s: abs(z_by_member[s])) if z_by_member else None
        max_z = abs(z_by_member.get(deviating, 0.0))
        return {
            "fired": bool(fired),
            "score": 1.0 - math.exp(-max_z / 2.0),
            "deviating": deviating,
            "expected": pred_by_member.get(deviating),  # scaled; the AnomalyDetectionGroup unscales it
            "recovered": max_z < _MULTI_RECOVERY_Z,
            "stats": {"z": round(max_z, 3), "deviating": deviating},
        }

    @property
    def detector(self):
        return self._models


@brick
class AnomalyDetectionGroup:
    """Watches how 2-5 related signals behave together, not just individually.

    Each member also runs its own full AnomalyDetection-equivalent scorer (per-member events use
    the "group.member" metric). watch="joint" learns which combinations of values are
    normal; watch="relationship" learns the usual coupling between the signals and fires
    when it breaks.

    Examples:
        >>> g = AnomalyDetectionGroup("rig", ["vibration", "power"], sensitivity="high")
        >>> g.on_anomaly(lambda e: print(e.summary or e.metric))
        >>> g.push(vibration=0.21, power=118.0)
    """

    def __init__(
        self,
        name: str,
        signals: list[str],
        *,
        watch: str = "joint",
        evaluate: str = "on_any",
        sensitivity="medium",
        period=None,
        limits=None,
        bucket=None,
        persist: bool = True,
    ):
        """Create a group over the given signal names.

        Member-level kwargs (period, limits, bucket, sensitivity) accept a scalar
        (every member, same value) or a list with exactly one slot per signal, where None
        means that member's default. AnomalyDetectionGroup-level kwargs (watch, evaluate, persist) never
        accept arrays.

        Args:
            name (str): AnomalyDetectionGroup name, used in events and as the persistence key.
            signals (list[str]): 2-5 member signal names.
            watch (str): "joint" (is this combination abnormal?) or "relationship"
                (has the usual coupling broken?).
            evaluate (str): "on_any" (default), "on_<signal>", "on_slowest", or
                "every:<duration>" for a fixed sample-and-hold timer.
            sensitivity: Member scalar/array; a scalar also tunes the joint score, an
                array leaves the joint score at "medium" (see joint_sensitivity).
            period: Member scalar/array of seasonal periods.
            limits: Member scalar/array of hard (lo, hi) bounds.
            bucket: Member scalar/array of seasonal rollup settings.
            persist (bool): Auto-save/restore model state across restarts.
        """
        if not name or not isinstance(name, str):
            raise ValueError("AnomalyDetectionGroup needs a non-empty name")
        if not isinstance(signals, (list, tuple)) or len(signals) < 2:
            raise ValueError(f"'{name}': an AnomalyDetectionGroup needs at least 2 signals; use AnomalyDetection for a single one")
        if len(signals) > MAX_GROUP_MEMBERS:
            raise ValueError(
                f"'{name}': an AnomalyDetectionGroup watches at most {MAX_GROUP_MEMBERS} signals; beyond that, bring your own "
                f"detector (e.g. river's anomaly.HalfSpaceTrees directly)"
            )
        if len(set(signals)) != len(signals):
            raise ValueError(f"'{name}': duplicate signal names")
        for kwarg, value in (("watch", watch), ("evaluate", evaluate), ("persist", persist)):
            reject_group_kwarg_array(kwarg, value)
        if watch not in ("joint", "relationship"):
            raise ValueError(f"'{name}': watch={watch!r} must be 'joint' or 'relationship'")

        self.name = name
        self.signals = list(signals)
        self.watch = watch
        self.persist = persist
        self._evaluate_intent = evaluate
        self._eval_mode, self._eval_arg = self._parse_evaluate(evaluate)

        # sensitivity special rule: a scalar (name or pro dict) tunes members and the
        # joint score; an array tunes members per element and the joint score stays at
        # "medium".
        joint_sens_name = "medium" if isinstance(sensitivity, list) else sensitivity
        member_sens = broadcast_member_kwarg("sensitivity", sensitivity, self.signals, "medium")
        member_period = broadcast_member_kwarg("period", period, self.signals, None)
        reject_flattened_limits("limits", limits)
        member_limits = broadcast_member_kwarg("limits", limits, self.signals, None)
        member_bucket = broadcast_member_kwarg("bucket", bucket, self.signals, "auto")

        self._members = {}
        for index, signal in enumerate(self.signals):
            self._members[signal] = SignalPipeline(
                f"{name}.{signal}",
                sensitivity=member_sens[index],
                period=member_period[index],
                limits=member_limits[index],
                bucket=member_bucket[index],
            )

        self._joint_sens_name = joint_sens_name
        self._joint_sens = resolve_sensitivity(joint_sens_name)
        self._gate = EpisodeGate(self._joint_sens.gate, self._joint_sens.hysteresis, self._joint_sens.score_floor)
        self._scales = {s: _ScaleWindow() for s in self.signals}
        if watch == "joint":
            self._watcher = _JointWatcher(self.signals, self._joint_sens.quantile)
        elif len(self.signals) == 2:
            self._watcher = _PairRelationship(self._joint_sens.ph_threshold)
        else:
            self._watcher = _MultiRelationship(self.signals, self._joint_sens.ph_threshold)

        self._lock = threading.RLock()
        self._callbacks = CallbackRegistry()
        self._dirty = False
        self._last_save = time.monotonic()
        self._last = {}  # signal -> (value, at), last-known-value join
        self._eval_times = deque(maxlen=64)
        self._evaluations = 0
        self._ready_announced = False
        self._stalled_episode = {s: False for s in self.signals}
        self._slowest = None
        self._last_timer_eval = None
        self.warmup_override = None

        if persist:
            restored = load_state(name, self._fingerprint())
            if isinstance(restored, dict):
                self._restore(restored)

        member_echo = " ".join(f"{s}[{self._members[s].echo()}]" for s in self.signals)
        print(f"{name}: {member_echo} · watch={watch} · evaluate={evaluate}", flush=True)

    def _parse_evaluate(self, evaluate: str) -> tuple[str, object]:
        if not isinstance(evaluate, str):
            raise ValueError(f"'{self.name}': evaluate must be a string")
        if evaluate == "on_any":
            return "any", None
        if evaluate == "on_slowest":
            return "slowest", None
        if evaluate == "every" or evaluate.startswith("every:"):
            _, _, duration = evaluate.partition(":")
            if not duration:
                raise ValueError(f'\'{self.name}\': evaluate="every" needs a duration, e.g. evaluate="every:30s"')
            return "every", parse_duration(duration, "evaluate")
        if evaluate.startswith("on_"):
            signal = evaluate[3:]
            if signal not in self.signals:
                raise ValueError(f"'{self.name}': evaluate={evaluate!r} does not match a declared signal ({', '.join(self.signals)})")
            return "signal", signal
        raise ValueError(f"'{self.name}': evaluate={evaluate!r} must be 'on_any', 'on_<signal>', 'on_slowest' or 'every:<duration>'")

    # ---- data ingestion ----------------------------------------------------------

    def push(self, values: dict | None = None, at: float | None = None, **kwargs):
        """Feed fresh values for one or more members; fires any resulting callbacks.

        Values passed in one call share one arrival time (exact alignment); separate
        calls get separate timestamps and last-known-value join applies at evaluation.

        Args:
            values (dict): Mapping of signal name to value; can also be given as kwargs.
            at (float): Optional arrival timestamp (epoch seconds), defaults to now.
            **kwargs: Signal values by name, merged with (and overriding) `values`.
        """
        merged = {**(values or {}), **kwargs}
        unknown = set(merged) - set(self.signals)
        if unknown:
            raise ValueError(f"'{self.name}': unknown signal(s) {', '.join(sorted(unknown))}; declared: {', '.join(self.signals)}")
        if not merged:
            raise ValueError(f"'{self.name}': push needs at least one value")
        at = time.time() if at is None else at

        events = []
        with self._lock:
            self._dirty = True
            for signal, value in merged.items():
                events.extend(self._members[signal].process(float(value), at))
                self._last[signal] = (float(value), at)
                self._scales[signal].add(float(value))
                self._stalled_episode[signal] = False
            self._maybe_pick_slowest()
            if self._should_evaluate(merged):
                events.extend(self._evaluate(at))
        for event in events:
            self._callbacks.dispatch(event)

    def _should_evaluate(self, pushed: dict) -> bool:
        if self._eval_mode == "any":
            return True
        if self._eval_mode == "signal":
            return self._eval_arg in pushed
        if self._eval_mode == "slowest":
            return True if self._slowest is None else self._slowest in pushed
        return False  # "every" evaluates from the timer, sample-and-hold

    def _maybe_pick_slowest(self):
        if self._eval_mode != "slowest" or self._slowest is not None:
            return
        if all(m.points >= CADENCE_RESOLUTION_POINTS for m in self._members.values()):
            self._slowest = max(self.signals, key=lambda s: self._members[s].cadence.median_iat or 0.0)

    # ---- evaluation --------------------------------------------------------------

    def _evaluation_interval(self) -> float | None:
        if self._eval_mode == "every":
            return self._eval_arg
        if len(self._eval_times) < 2:
            return None
        times = list(self._eval_times)
        return statistics.median(b - a for a, b in zip(times, times[1:]))

    def _evaluate(self, at: float) -> list[AnomalyEvent]:
        interval = self._evaluation_interval()
        self._eval_times.append(at)
        if len(self._last) < len(self.signals):
            return []  # cannot join yet, some member never pushed

        stale, events = self._check_staleness(at, interval)
        if stale:
            return events  # a stale required member: skip joint scoring

        self._evaluations += 1
        values = {s: self._last[s][0] for s in self.signals}
        if self.watch == "joint":
            scaled = {s: self._scales[s].scale(values[s]) for s in self.signals}
            score, anomalous = self._watcher.evaluate(scaled)
            events.extend(self._announce_ready(values, at))
            if self.ready:
                events.extend(self._judge_joint(values, score, anomalous, at, interval))
        else:
            result = (
                self._watcher.evaluate(values[self.signals[0]], values[self.signals[1]])
                if isinstance(self._watcher, _PairRelationship)
                else self._watcher.evaluate({s: self._scales[s].scale(values[s]) for s in self.signals})
            )
            events.extend(self._announce_ready(values, at))
            if self.ready:
                events.extend(self._judge_relationship(values, result, at))
        return events

    def _check_staleness(self, at: float, interval: float | None) -> tuple[bool, list[AnomalyEvent]]:
        """Returns (any member stale, on_stalled events for members entering a staleness episode)."""
        stale_found = False
        events = []
        for signal in self.signals:
            member = self._members[signal]
            median_iat = member.cadence.median_iat
            if median_iat is None and interval is None:
                continue
            threshold = max(3 * (median_iat or 0.0), 2 * (interval or 0.0))
            if threshold <= 0 or at - self._last[signal][1] <= threshold:
                continue
            stale_found = True
            if not self._stalled_episode[signal]:
                self._stalled_episode[signal] = True
                events.append(
                    AnomalyEvent(
                        metric=f"{self.name}.{signal}",
                        kind="stalled",
                        value=self._last[signal][0],
                        expected=None,
                        score=0.0,
                        at=at,
                        stats=Stats(),
                    )
                )
        return stale_found, events

    def _announce_ready(self, values: dict, at: float) -> list[AnomalyEvent]:
        if not self.ready or self._ready_announced:
            return []
        self._ready_announced = True
        return [AnomalyEvent(metric=self.name, kind="ready", value=dict(values), expected=None, score=0.0, at=at, stats=Stats())]

    def _judge_joint(self, values: dict, score: float, anomalous: bool, at: float, interval: float | None) -> list[AnomalyEvent]:
        agreement, lead, window = self._coincidence(at, interval)
        coherent = len(agreement) >= 2
        gate = self._joint_sens.gate if coherent else raise_gate(self._joint_sens.gate)
        verdict = self._gate.observe(anomalous, score, gate=gate)
        if verdict is None:
            return []
        expected = {s: self._members[s].last_expected for s in self.signals}
        stats = Stats(agreement=agreement, lead=lead, coherent=coherent, window=round(window, 3), detector=self._watcher.detector)
        summary = self._summary(agreement) if verdict == "anomaly" else None
        return [AnomalyEvent(metric=self.name, kind=verdict, value=dict(values), expected=expected, score=score, at=at, stats=stats, summary=summary)]

    def _coincidence(self, at: float, interval: float | None) -> tuple[list[str], dict, float]:
        """Which members' own scores exceeded their own cutoff within the coincidence window."""
        slowest_iat = max((self._members[s].cadence.median_iat or 0.0) for s in self.signals)
        window = max(2 * slowest_iat, interval or 0.0) or 1.0
        exceeded = {
            s: self._members[s].last_exceeded_at
            for s in self.signals
            if self._members[s].last_exceeded_at is not None and at - self._members[s].last_exceeded_at <= window
        }
        agreement = sorted(exceeded, key=exceeded.get)
        first = min(exceeded.values()) if exceeded else None
        lead = {s: round(exceeded[s] - first, 3) for s in agreement}
        return agreement, lead, window

    def _summary(self, agreement: list[str]) -> str:
        if len(agreement) >= 2:
            return f"{self.name} anomaly ({' + '.join(agreement)} agree)"
        if len(agreement) == 1:
            return f"{agreement[0]} only — other signals normal (possible sensor issue)"
        return f"{self.name} anomaly (combination unusual — no single signal stands out)"

    def _judge_relationship(self, values: dict, result: dict, at: float) -> list[AnomalyEvent]:
        expected = result["expected"]
        if result["deviating"] is not None and expected is not None:
            expected = self._scales[result["deviating"]].unscale(expected)
        stats = Stats(result["stats"], detector=self._watcher.detector)

        if not self._gate.in_anomaly:
            if result["fired"] and result["score"] >= self._joint_sens.score_floor:
                self._gate.force_anomalous()
                return [
                    AnomalyEvent(metric=self.name, kind="anomaly", value=dict(values), expected=expected, score=result["score"], at=at, stats=stats)
                ]
            return []
        if self._gate.observe(not result["recovered"], result["score"]) == "normal":
            return [AnomalyEvent(metric=self.name, kind="normal", value=dict(values), expected=expected, score=result["score"], at=at, stats=stats)]
        return []

    # ---- callbacks ---------------------------------------------------------------

    def on_anomaly(self, callback: callable):
        """Register a callback for anomaly events (joint, relationship and per-member)."""
        self._callbacks.register("anomaly", callback)

    def on_normal(self, callback: callable):
        """Register a callback for recovery events after an anomaly episode."""
        self._callbacks.register("normal", callback)

    def on_drift(self, callback: callable):
        """Register a callback for per-member level-shift events."""
        self._callbacks.register("drift", callback)

    def on_stalled(self, callback: callable):
        """Register a callback fired when a member's feed goes silent."""
        self._callbacks.register("stalled", callback)

    def on_ready(self, callback: callable):
        """Register a callback fired once when the group's warm-up completes."""
        self._callbacks.register("ready", callback)

    # ---- introspection -----------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the joint/relationship warm-up is satisfied."""
        if self.warmup_override is not None:
            return self._evaluations >= self.warmup_override
        mult = self._joint_sens.warmup_mult
        if self.watch == "joint":
            members_ready = all(m.ready for m in self._members.values())
            return members_ready and self._evaluations >= math.ceil(GROUP_JOINT_WARMUP * mult)
        return self._evaluations >= math.ceil(GROUP_RELATIONSHIP_WARMUP * mult)

    @property
    def warmup(self) -> int | None:
        """Warm-up override in evaluations; None means the derived formula applies."""
        return self.warmup_override

    @warmup.setter
    def warmup(self, evaluations: int):
        with self._lock:
            self.warmup_override = int(evaluations)

    @property
    def joint_sensitivity(self) -> str:
        """Sensitivity of the joint score; set before pushing data (pro override)."""
        return self._joint_sens_name

    @joint_sensitivity.setter
    def joint_sensitivity(self, name: str):
        with self._lock:
            self._joint_sens = resolve_sensitivity(name, "joint_sensitivity")
            self._joint_sens_name = name
            self._gate = EpisodeGate(self._joint_sens.gate, self._joint_sens.hysteresis, self._joint_sens.score_floor)
            if isinstance(self._watcher, _JointWatcher):
                self._watcher.set_quantile(self._joint_sens.quantile)
            else:
                self._watcher.set_threshold(self._joint_sens.ph_threshold)

    @property
    def stats(self) -> Stats:
        """Live resolved view of the group configuration and runtime state."""
        with self._lock:
            return Stats(
                watch=self.watch,
                evaluate=self._evaluate_intent,
                ready=self.ready,
                evaluations=self._evaluations,
                slowest=self._slowest,
                members={s: Stats(self._members[s].intent, ready=self._members[s].ready) for s in self.signals},
                detector=self._watcher.detector,
            )

    def recalibrate(self):
        """Adopt the current regime as the new normal, on demand.

        Every member re-learns from incoming data (warm-up state kept) and the joint /
        relationship model starts fresh, re-warming before it judges again. Use after an
        intentional process change that should not stay flagged as anomalous.
        """
        with self._lock:
            for member in self._members.values():
                member.recalibrate()
            if self.watch == "joint":
                self._watcher = _JointWatcher(self.signals, self._joint_sens.quantile)
            elif len(self.signals) == 2:
                self._watcher = _PairRelationship(self._joint_sens.ph_threshold)
            else:
                self._watcher = _MultiRelationship(self.signals, self._joint_sens.ph_threshold)
            self._gate = EpisodeGate(self._joint_sens.gate, self._joint_sens.hysteresis, self._joint_sens.score_floor)
            self._evaluations = 0
            self._ready_announced = False
            self._dirty = True

    # ---- lifecycle ---------------------------------------------------------------

    def loop(self):
        """Framework loop: fixed-timer evaluations and periodic auto-save."""
        time.sleep(0.2)
        events = []
        if self._eval_mode == "every":
            now = time.time()
            with self._lock:
                if self._last and (self._last_timer_eval is None or now - self._last_timer_eval >= self._eval_arg):
                    self._last_timer_eval = now
                    events = self._evaluate(now)
        for event in events:
            self._callbacks.dispatch(event)
        if time.monotonic() - self._last_save >= AUTOSAVE_INTERVAL_S:
            self._save_if_dirty()

    def stop(self):
        """Persist state on clean shutdown."""
        self._save_if_dirty()

    # ---- persistence -------------------------------------------------------------

    def _fingerprint(self) -> str:
        fields = {
            "watch": self.watch,
            "evaluate": self._evaluate_intent,
            # The resolved internals, not the intent value: deterministic for dicts too.
            "joint_sensitivity": astuple(self._joint_sens),
            "signals": tuple(self.signals),
        }
        for signal, member in self._members.items():
            fields[f"member:{signal}"] = tuple(sorted(member.fingerprint_fields().items(), key=lambda kv: kv[0]))
        return config_fingerprint(fields)

    def _state(self) -> dict:
        return {
            "members": self._members,
            "scales": self._scales,
            "watcher": self._watcher,
            "gate": self._gate,
            "last": self._last,
            "evaluations": self._evaluations,
            "ready_announced": self._ready_announced,
            "slowest": self._slowest,
        }

    def _restore(self, state: dict):
        self._members = state["members"]
        self._scales = state["scales"]
        self._watcher = state["watcher"]
        self._gate = state["gate"]
        self._last = state["last"]
        self._evaluations = state["evaluations"]
        self._ready_announced = state["ready_announced"]
        self._slowest = state["slowest"]

    def _save_if_dirty(self):
        if not self.persist:
            return
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            save_state(self.name, self._fingerprint(), self._state())
        self._last_save = time.monotonic()
