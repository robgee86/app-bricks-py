# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""One signal's full detection pipeline: limits -> learned detector -> drift, plus forecast."""

import math
import statistics

from .config import (
    BASE_WARMUP_POINTS,
    CADENCE_RESOLUTION_POINTS,
    DRIFT_WARMUP_POINTS,
    FLUSH_CONFIRM_POINTS,
    FLUSH_CONFIRM_SIGMAS,
    config_fingerprint,
    parse_duration,
    raise_gate,
    resolve_sensitivity,
    validate_limits,
)
from .detectors import (
    Bucket,
    CadenceTracker,
    DriftPath,
    ForecastRunner,
    LimitsGuard,
    RateBucketer,
    SeasonalPath,
    TrendForecaster,
    TrimmedPath,
)
from .events import AnomalyEvent, EpisodeGate, Stats, logger

# Intent kwargs that tune the learned detector and are therefore ignored with detector=.
_DETECTOR_TUNING_KWARGS = ("period", "model", "bucket")


class ByoPath:
    """Bring-your-own river detector, wrapped in a QuantileFilter unless it already filters.

    The detector must accept score_one(None, value) / learn_one(None, value).
    """

    def __init__(self, detector, quantile: float):
        from .detectors import quantile_filter

        if hasattr(detector, "classify"):
            self._filter = detector
        else:
            self._filter = quantile_filter(detector, quantile)

    def evaluate(self, value: float):
        from .detectors import Evaluation

        score = self._filter.score_one(None, value)
        anomalous = self._filter.classify(score)
        self._filter.learn_one(None, value)
        return Evaluation(score, anomalous, None, {})

    @property
    def detector(self):
        return self._filter


class SignalPipeline:
    """Resolves one signal's intent kwargs into detectors and evaluates pushes through them.

    Layering per push: limits -> learned detector -> drift; the first layer to fire
    determines the event, the others still update state. Callbacks are not handled here:
    process() returns events for the owning AnomalyDetection/AnomalyDetectionGroup to dispatch.
    """

    def __init__(
        self,
        metric: str,
        *,
        sensitivity="medium",
        period: str | None = None,
        limits: tuple | None = None,
        bucket="auto",
        model: str | None = None,
        forecast: str | None = None,
        drift: str = "page_hinkley",
        detector=None,
    ):
        self.metric = metric
        self.intent = {
            "sensitivity": sensitivity,
            "period": period,
            "limits": limits,
            "bucket": bucket,
            "model": model,
            "forecast": forecast,
        }
        self.sens = resolve_sensitivity(sensitivity)
        self.sensitivity = sensitivity

        # Combination matrix: everything invalid raises at construction.
        if bucket != "auto" and period is None:
            raise ValueError(f"'{metric}': bucket only exists for the seasonal path; it requires period")
        if model is not None and model != "sarimax":
            raise ValueError(f"'{metric}': model={model!r} is not supported (only 'sarimax')")
        if model == "sarimax" and period is None:
            raise ValueError(f"'{metric}': model='sarimax' is seasonal and requires period")
        if forecast is not None and limits is None:
            raise ValueError(f"'{metric}': a forecast breach needs a bound to breach; set limits")
        if drift not in ("page_hinkley", "adwin"):
            raise ValueError(f"'{metric}': drift={drift!r} must be 'page_hinkley' or 'adwin'")

        self.period_s = parse_duration(period, "period") if period is not None else None
        bucket_resolved = bucket if bucket in ("auto", "off") else parse_duration(bucket, "bucket")
        self.limits = validate_limits(limits) if limits is not None else None
        self.forecast_s = parse_duration(forecast, "forecast") if forecast is not None else None

        self._guard = LimitsGuard(self.limits) if self.limits else None
        self._seasonal = None
        self._trend = None
        self.is_byo = detector is not None
        if self.is_byo:
            provided = [k for k in _DETECTOR_TUNING_KWARGS if self.intent[k] not in (None, False, "auto")]
            if provided:
                logger.warning(f"'{metric}': detector= replaces the learned detector; ignoring {', '.join(provided)}")
            self._learned = ByoPath(detector, self.sens.quantile)
        elif self.period_s is not None:
            self._seasonal = SeasonalPath(self.period_s, self.sens.quantile, model=model or "hw", bucket=bucket_resolved)
            self._learned = self._seasonal
        else:
            self._learned = TrimmedPath(self.sens.quantile)
        if self.forecast_s is not None and self._seasonal is None:
            self._trend = TrendForecaster()

        self._drift = DriftPath(drift, self.sens.ph_threshold, DRIFT_WARMUP_POINTS)
        self._forecast_runner = ForecastRunner(self.forecast_s) if self.forecast_s is not None else None
        self._gate = EpisodeGate(self.sens.gate, self.sens.hysteresis, self.sens.score_floor)

        self.cadence = CadenceTracker()
        self.points = 0
        self.learned_evaluations = 0
        self.warmup_override = None
        self._bucketer = RateBucketer()
        self._spread_path = None  # created if the rate guard trips
        self._centroid_path = None  # created if the rate guard trips with enough samples per bucket
        self._iat_fixed = None  # raw cadence a resolved granularity was derived from
        self._ready_announced = False
        self._cadence_flagged = False
        self._pending_flush = None
        self._stalled = False
        self.last_value = None
        self.last_expected = None
        self.last_exceeded_at = None  # when this signal's own score last exceeded its own cutoff

    # ---- warm-up ----------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the learned detector's warm-up is satisfied (limits never wait).

        Warm-up counts evaluations, not pushes: under the rate guard those are buckets,
        so warm-up means the same wall-clock time at any push rate.
        """
        if self.warmup_override is not None:
            return self.learned_evaluations >= self.warmup_override
        if self._seasonal is not None:
            if not self._seasonal.resolved:
                return False
            # M_sens never shortens the seasonal warm-up below 2 periods; low-style
            # tuning (multiplier >= low's) extends it to 3.
            periods = 3 if self.sens.warmup_mult >= 1.5 else 2
            return self._seasonal.model_updates >= periods * self._seasonal.L
        return self.learned_evaluations >= math.ceil(BASE_WARMUP_POINTS * self.sens.warmup_mult)

    # ---- evaluation -------------------------------------------------------------

    def process(self, value: float, at: float) -> list[AnomalyEvent]:
        """Evaluate one push through every layer; returns the events to dispatch."""
        events = []
        self.cadence.update(at)
        self.points += 1
        self.last_value = value
        self._stalled = False

        self._maybe_resolve()

        # Layer 1: hard limits check every raw push, live from the first one.
        limit_violated = False
        if self._guard is not None:
            limit_violated, bound = self._guard.check(value)
            if limit_violated and not self._gate.in_anomaly:
                self._gate.force_anomalous()
                events.append(self._event("anomaly", value, bound, 1.0, at, Stats(reason="limit")))

        # Layers 2 and 3 run at scoring granularity: per push, or per completed bucket
        # under the rate guard; they keep learning even when the limits layer fired first.
        primary, anomalous, lane = None, False, None
        expected_level = None
        drift_fired = False
        bucket = self._bucketer.update(value)
        if bucket is not None:
            primary, anomalous, lane = self._evaluate_bucket(bucket, at)
            expected_level = self._drift.expected
            drift_fired = self._drift.update(bucket.level) and self._drift.updates >= DRIFT_WARMUP_POINTS
            if self._trend is not None:
                self._trend.learn(bucket.level)

        if self.ready and not self._ready_announced:
            self._ready_announced = True
            events.append(self._event("ready", value, self.last_expected, 0.0, at, Stats()))

        # Exactly one gate observation per push: an active limit breach holds the episode
        # open; otherwise the learned verdict rules once ready; before that, in-bounds
        # pushes still tick the recovery hysteresis so a limit episode can close.
        learned_fired = False
        if limit_violated:
            self._gate.observe(True, 1.0)
        elif primary is not None and self.ready:
            # Dispersion and pitch evidence is bursty by nature: anomalies driven by
            # those lanes must sustain one gate step longer, so a brief handling burst
            # stays silent.
            gate_override = raise_gate(self.sens.gate) if lane in ("spread", "spectrum") and anomalous else None
            verdict = self._gate.observe(anomalous, primary.score, gate=gate_override)
            if verdict is not None:
                learned_fired = verdict == "anomaly"
                events.append(self._event(verdict, value, primary.expected, primary.score, at, self._lane_stats(primary, lane)))
        elif not self.ready and self._gate.in_anomaly:
            if self._gate.observe(False, 0.0) == "normal":
                events.append(self._event("normal", value, self.last_expected, 0.0, at, Stats()))

        # Layer 3 events: an earlier layer's event wins the push.
        if drift_fired and not limit_violated and not learned_fired:
            events.append(self._event("drift", value, expected_level, 1.0, at, Stats(detector=self._drift.detector)))
        if bucket is not None:
            self._advance_flush(bucket.level, drift_fired)
        events.extend(self._check_cadence(value, at))

        if self._forecast_runner is not None and self.ready and bucket is not None:
            events.extend(self._run_forecast(value, at))
        return events

    def _evaluate_bucket(self, bucket: Bucket, at: float):
        """Judge both faces of the bucket; returns (primary evaluation, anomalous, lane).

        A bucket's mean is only known to +/- its standard error (spread / sqrt(k)), so
        the level lane scores against it: an oscillation's phase leakage into the mean
        is not evidence of a level change.
        """
        if self._seasonal is not None:
            evaluation = self._seasonal.update(bucket.level)
        elif isinstance(self._learned, TrimmedPath) and bucket.spread is not None:
            evaluation = self._learned.evaluate(bucket.level, uncertainty=bucket.spread / math.sqrt(self._bucketer.k))
        else:
            evaluation = self._learned.evaluate(bucket.level)
        if evaluation is not None:
            self.learned_evaluations += 1
            self.last_expected = evaluation.expected
        # Episode-scoped protection for the unprotected lanes: learning everything is
        # what lets recurring ordinary activity become normal, but while an anomaly
        # episode is open the fault itself must not be absorbed into the band.
        learn = not self._gate.in_anomaly
        lanes = [(evaluation, "level")] if evaluation is not None else []
        if self._spread_path is not None and bucket.spread is not None:
            lanes.append((self._spread_path.evaluate(bucket.spread, learn=learn), "spread"))
        if self._centroid_path is not None and bucket.centroid is not None:
            lanes.append((self._centroid_path.evaluate(bucket.centroid, learn=learn), "spectrum"))
        if not lanes:
            return None, False, None

        # The anomalous lane drives reporting; on ties, the higher score does.
        anomalous_lanes = [entry for entry in lanes if entry[0].anomalous]
        primary, lane = max(anomalous_lanes or lanes, key=lambda entry: entry[0].score)
        anomalous = bool(anomalous_lanes)
        if anomalous:
            self.last_exceeded_at = at
        return primary, anomalous, (lane if self._spread_path is not None else None)

    def _lane_stats(self, primary, lane: str | None) -> Stats:
        paths = {"spread": self._spread_path, "spectrum": self._centroid_path}
        detector = (paths.get(lane) or self._learned).detector
        stats = Stats(primary.stats, detector=detector)
        if lane is not None:
            stats["lane"] = lane
        if self.bucket_size_s is not None:
            stats["bucket_size"] = self.bucket_size_s
        return stats

    def _maybe_resolve(self):
        """Fix the scoring granularity once the cadence is measured: rate guard, then L."""
        if self._bucketer.resolved or self.points < CADENCE_RESOLUTION_POINTS:
            return
        median_iat = self.cadence.median_iat
        if median_iat is None:
            return
        self._bucketer.resolve(median_iat)
        if self._bucketer.active:
            # No protection: bucket spreads and centroids are legitimately bimodal
            # (quiet vs active), and both modes must be learnable or ordinary activity
            # stays anomalous forever.
            self._spread_path = TrimmedPath(self.sens.quantile, protect=False)
            self._centroid_path = TrimmedPath(self.sens.quantile, protect=False)
            if self._seasonal is not None:
                self._seasonal.reset_buffer()
            logger.warning(
                f"'{self.metric}': sampling ~{median_iat * 1000:.3g}ms -> scoring on {self._bucketer.bucket_size_s:.3g}s buckets "
                f"(level + spread + pitch); learned events reflect sustained behavior, not single samples. "
                f"Hard limits still check every push."
            )
        if self._seasonal is not None:
            self._seasonal.resolve(self._bucketer.bucket_size_s or median_iat)
            if self._seasonal.guard_tripped:
                logger.warning(
                    f"'{self.metric}': sampling ~{median_iat:.3g}s with period={self.intent['period']!r} -> seasonal scoring on "
                    f"{self._seasonal.bucket_size_s:.3g}s averages; spikes shorter than ~{self._seasonal.bucket_size_s:.3g}s may be "
                    f"smoothed out. Hard limits and push rate are unaffected."
                )
        if self._bucketer.active or self._seasonal is not None:
            self._iat_fixed = median_iat

    def _advance_flush(self, value: float, drift_fired: bool):
        """Flush the learned window after a drift alarm, but only once the shift is confirmed.

        A lone spike trips PageHinkley just like a step does; the difference only shows
        afterwards. So the flush is armed on the alarm and executed only if the median of
        the next FLUSH_CONFIRM_POINTS pushes is still far from the old center — flushing
        on a spike would adopt it as the new normal and silence the anomaly.
        """
        if self._pending_flush is None:
            if drift_fired and hasattr(self._learned, "flush"):
                mu, sigma = self._learned.moments()
                if mu is not None and sigma > 0:
                    self._pending_flush = {"mu": mu, "sigma": sigma, "recent": [value]}
            return
        recent = self._pending_flush["recent"]
        recent.append(value)
        if len(recent) < FLUSH_CONFIRM_POINTS:
            return
        pending, self._pending_flush = self._pending_flush, None
        if abs(statistics.median(recent) - pending["mu"]) > FLUSH_CONFIRM_SIGMAS * pending["sigma"]:
            self._learned.flush()
            for path in (self._spread_path, self._centroid_path):
                if path is not None:
                    path.flush()

    def _check_cadence(self, value: float, at: float) -> list[AnomalyEvent]:
        ewma, fixed = self.cadence.ewma_iat, self._iat_fixed
        if not ewma or not fixed:
            return []
        drifted = ewma > 2 * fixed or ewma < fixed / 2
        if drifted and not self._cadence_flagged:
            self._cadence_flagged = True
            return [self._event("drift", value, None, 1.0, at, Stats(reason="cadence_change"))]
        if not drifted:
            self._cadence_flagged = False
        return []

    def _run_forecast(self, value: float, at: float) -> list[AnomalyEvent]:
        source = self._seasonal if self._seasonal is not None else self._trend
        step_s = self.bucket_size_s or self.cadence.median_iat
        steps = self._forecast_runner.horizon_steps(step_s)
        breach = self._forecast_runner.evaluate(source.forecast(steps), step_s, self._guard)
        if breach is None:
            return []
        stats = Stats(eta_s=round(breach["eta_s"], 3), detector=source.detector)
        return [self._event("forecast", value, breach["projected"], 1.0, at, stats)]

    def recalibrate(self):
        """Forget learned normality and re-learn from incoming data (on-demand adoption).

        Bands and score cutoffs start fresh and the warm-up restarts (a fresh band must
        mature before it judges again); cadence and bucketing resolution are kept. An
        open anomaly episode closes with a genuine on_normal within a few pushes.
        """
        for path in (self._learned, self._spread_path, self._centroid_path):
            recalibrate = getattr(path, "recalibrate", None)
            if recalibrate is not None:
                recalibrate()
        self._pending_flush = None
        self.learned_evaluations = 0
        self._ready_announced = False

    def check_staleness(self, now: float) -> list[AnomalyEvent]:
        """Emit on_stalled once per staleness episode; a fresh push resets the episode."""
        last_at = self.cadence.last_at
        if self._stalled or last_at is None:
            return []
        median_iat = self.cadence.median_iat
        threshold = max(5 * median_iat, 30.0) if median_iat else 30.0
        if now - last_at <= threshold:
            return []
        self._stalled = True
        return [self._event("stalled", self.last_value, self.last_expected, 0.0, now, Stats())]

    def _event(self, kind, value, expected, score, at, stats) -> AnomalyEvent:
        return AnomalyEvent(metric=self.metric, kind=kind, value=value, expected=expected, score=score, at=at, stats=stats)

    # ---- introspection ----------------------------------------------------------

    @property
    def bucket_size_s(self) -> float | None:
        """Effective learned-scoring granularity in seconds, None when scoring raw pushes."""
        if self._seasonal is not None and self._seasonal.bucket_size_s is not None:
            return self._seasonal.bucket_size_s
        return self._bucketer.bucket_size_s

    @property
    def detector(self):
        """The live learned detector (pro escape hatch)."""
        return self._learned.detector

    def echo(self) -> str:
        """Intent-level resolved config, e.g. "period=1d, sensitivity=medium"."""
        parts = []
        for key in ("period", "limits", "bucket", "model", "forecast"):
            value = self.intent[key]
            if value in (None, False) or (key == "bucket" and value == "auto"):
                continue
            parts.append(f"{key}={value if not isinstance(value, str) else value}")
        parts.append(f"sensitivity={self.sensitivity}")
        return ", ".join(parts)

    def fingerprint_fields(self) -> dict:
        """Resolved internals whose change must invalidate persisted state."""
        return {
            "detector": type(self._learned).__name__,
            "quantile": self.sens.quantile,
            "gate": self.sens.gate,
            "hysteresis": self.sens.hysteresis,
            "score_floor": self.sens.score_floor,
            "warmup_mult": self.sens.warmup_mult,
            "ph_threshold": self.sens.ph_threshold,
            "period_s": self.period_s,
            "limits": self.limits,
            "bucket": str(self.intent["bucket"]),
            "model": self.intent["model"],
            "forecast_s": self.forecast_s,
            "drift": self._drift.kind,
        }

    def fingerprint(self) -> str:
        return config_fingerprint(self.fingerprint_fields())
