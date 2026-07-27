# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Internal detector layer: the only module that touches river classes directly."""

import math
import statistics
from collections import deque
from dataclasses import dataclass

from river import anomaly, drift, stats as river_stats, time_series

from .config import MAX_SEASONAL_LENGTH, RATE_BUCKET_TARGET_S, RATE_GUARD_IAT_S, TARGET_SEASONAL_LENGTH

# Holt-Winters smoothing constants; the batch re-fit owns these, never the sensitivity dial.
HW_ALPHA, HW_BETA, HW_GAMMA = 0.3, 0.1, 0.6

# The unified scorer's resolved internals. Behavior is flat across a wide range of these
# (validated 200-1500 / 5-20%): the adaptive QuantileFilter cutoff absorbs score
# miscalibration, so they only need ballpark correctness.
TRIM_WINDOW = 500
TRIM_ALPHA = 0.10
SCORER_GRACE = 30
# The QuantileFilter cutoff is estimated over a rolling score window: past anomaly
# bursts stop raising the bar once they leave it (an unbounded quantile ratchets toward
# 1.0 under repeated anomalies and never readmits moderate ones).
CUTOFF_WINDOW = 500


def quantile_filter(scorer, quantile: float, protect: bool = True) -> anomaly.QuantileFilter:
    """A QuantileFilter whose cutoff has bounded memory (rolling quantile)."""
    filter_ = anomaly.QuantileFilter(scorer, q=quantile, protect_anomaly_detector=protect)
    filter_.quantile = river_stats.RollingQuantile(q=quantile, window_size=CUTOFF_WINDOW)
    return filter_


def _trim_correction(alpha: float) -> float:
    """Consistency factor: the trimmed std of a Gaussian underestimates sigma by a known amount."""
    a = statistics.NormalDist().inv_cdf(1 - alpha)
    phi = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    return 1 / math.sqrt(1 - 2 * a * phi / (1 - 2 * alpha))


class CadenceTracker:
    """Measures inter-arrival time: EWMA (alpha=0.1) plus a rolling median."""

    def __init__(self):
        self._last_at = None
        self._ewma = None
        self._recent = deque(maxlen=512)

    def update(self, at: float):
        if self._last_at is not None:
            iat = max(at - self._last_at, 1e-9)
            self._ewma = iat if self._ewma is None else 0.1 * iat + 0.9 * self._ewma
            self._recent.append(iat)
        self._last_at = at

    @property
    def last_at(self) -> float | None:
        return self._last_at

    @property
    def ewma_iat(self) -> float | None:
        return self._ewma

    @property
    def median_iat(self) -> float | None:
        return statistics.median(self._recent) if self._recent else None


class LimitsGuard:
    """Hard bounds, live from the first push: no warm-up, no learned state."""

    def __init__(self, limits: tuple[float | None, float | None]):
        self.lo, self.hi = limits

    def check(self, value: float) -> tuple[bool, float]:
        """Returns (violated, nearest bound)."""
        if self.lo is not None and value < self.lo:
            return True, self.lo
        if self.hi is not None and value > self.hi:
            return True, self.hi
        return False, value

    def crossed(self, value: float) -> bool:
        return self.check(value)[0]


@dataclass
class Evaluation:
    """One learned-detector evaluation of one (possibly bucketed) point."""

    score: float
    anomalous: bool
    expected: float | None
    stats: dict


@dataclass(frozen=True)
class Bucket:
    """One scoring unit: a raw push, or an aggregate of them under the rate guard.

    High-rate kinematic signals put activity in the variance, not the level (the mean of
    an oscillation is ~0), so a bucket carries both faces of its samples.
    """

    level: float
    spread: float | None  # None when the bucket is a single raw push


class RateBucketer:
    """Aggregates high-rate pushes into ~1s buckets so scoring keeps human time scales.

    Below the guard rate this is a pass-through: every push is its own bucket.
    """

    def __init__(self):
        self.k = 1  # pushes per bucket
        self.bucket_size_s = None
        self.resolved = False
        self._acc = []

    def resolve(self, median_iat: float):
        self.resolved = True
        if median_iat < RATE_GUARD_IAT_S:
            self.k = max(2, round(RATE_BUCKET_TARGET_S / median_iat))
            self.bucket_size_s = round(self.k * median_iat, 3)

    @property
    def active(self) -> bool:
        return self.k > 1

    def update(self, value: float) -> Bucket | None:
        """Feed one raw push; returns a Bucket when one completes, None mid-bucket."""
        if not self.active:
            return Bucket(level=value, spread=None)
        self._acc.append(value)
        if len(self._acc) < self.k:
            return None
        mean = sum(self._acc) / len(self._acc)
        variance = sum((v - mean) ** 2 for v in self._acc) / len(self._acc)
        self._acc = []
        return Bucket(level=mean, spread=math.sqrt(max(variance, 0.0)))


class TrimmedScorer(anomaly.base.SupervisedAnomalyDetector):
    """Rolling trimmed-statistics scorer: one detector that stays safe across most feeds.

    Trimmed mean/sigma over a rolling window resist outlier contamination up to the trim
    rate while staying nearly as efficient as plain Gaussian statistics on clean data;
    the score is the Gaussian CDF rank under those statistics. The scale never drops
    below the signal's measured resolution, so quantized feeds (integer-step sensors)
    cannot produce a hair-trigger band.

    observe() must see every raw push: the resolution estimate would otherwise be
    starved by the QuantileFilter's protection refusing to learn outliers.
    """

    def __init__(self, window: int = TRIM_WINDOW, trim: float = TRIM_ALPHA, grace_period: int = SCORER_GRACE):
        self.window = window
        self.trim = trim
        self.grace_period = grace_period
        self._correction = _trim_correction(trim)
        self._values = deque(maxlen=window)
        self._diffs = deque(maxlen=200)
        self._last_raw = None
        # Transient per-evaluation measurement uncertainty (e.g. a bucket mean's standard
        # error): deviations within it are not evidence of change.
        self.uncertainty = 0.0

    def observe(self, y: float):
        """Track the raw stream's resolution; sees every push, learned or not."""
        if self._last_raw is not None and y != self._last_raw:
            self._diffs.append(abs(y - self._last_raw))
        self._last_raw = y

    def moments(self) -> tuple[float | None, float]:
        """Trimmed mean and consistency-corrected, resolution-floored sigma."""
        if not self._values:
            return None, 0.0
        ordered = sorted(self._values)
        cut = int(len(ordered) * self.trim)
        core = ordered[cut : len(ordered) - cut] or ordered
        mu = sum(core) / len(core)
        variance = sum((v - mu) ** 2 for v in core) / len(core)
        resolution = statistics.median(self._diffs) if self._diffs else 0.0
        return mu, max(math.sqrt(variance) * self._correction, resolution, 1e-12)

    def learn_one(self, x, y):
        self._values.append(y)

    def score_one(self, x, y):
        if len(self._values) < self.grace_period:
            return 0.0
        mu, sigma = self.moments()
        sigma = math.sqrt(sigma**2 + self.uncertainty**2)
        return 2 * abs(statistics.NormalDist(mu, sigma).cdf(y) - 0.5)

    def flush(self):
        """Forget the learned window: a confirmed level shift invalidated the old regime.

        The window must be cleared, not truncated: drift confirmation arrives points
        after the shift, so any kept tail is old-regime data, and the QuantileFilter's
        protection would never admit the new regime (its scores stay above the cutoff).
        An empty window re-enters the grace period, during which everything is learned,
        so the band re-centers on the new normal within grace_period points.
        """
        self._values.clear()


class TrimmedPath:
    """Adaptive non-seasonal detector: the trimmed scorer wrapped in a QuantileFilter.

    protect=False makes the scorer learn anomalous points too: right for recurring,
    legitimately bimodal statistics (bucket spreads), where refusing to learn the upper
    mode would keep it anomalous forever.
    """

    def __init__(self, quantile: float, protect: bool = True):
        self._scorer = TrimmedScorer()
        self._filter = quantile_filter(self._scorer, quantile, protect=protect)

    def evaluate(self, value: float, uncertainty: float = 0.0) -> Evaluation:
        self._scorer.observe(value)
        self._scorer.uncertainty = uncertainty
        mu, sigma = self._scorer.moments()
        sigma = math.sqrt(sigma**2 + uncertainty**2)
        z = (value - mu) / sigma if mu is not None and sigma > 0 else 0.0
        score = self._filter.score_one(None, value)
        anomalous = self._filter.classify(score)
        self._filter.learn_one(None, value)
        self._scorer.uncertainty = 0.0
        return Evaluation(score, anomalous, mu, {"z": z, "threshold": self._filter.quantile.get()})

    def flush(self):
        self._scorer.flush()

    def moments(self) -> tuple[float | None, float]:
        return self._scorer.moments()

    @property
    def detector(self):
        return self._filter


class SeasonalPath:
    """Seasonal detector: HW/SNARIMAX residuals scored through a QuantileFilter.

    The seasonal length L is unknown until the push cadence has been measured, so values
    are buffered until resolve() constructs the model and replays them. The rollup guard
    aggregates pushes into buckets when L would be unmanageable.
    """

    def __init__(self, period_s: float, quantile: float, model: str = "hw", bucket="auto"):
        self.period_s = period_s
        self.model_kind = model
        self.bucket_mode = bucket
        self._quantile = quantile
        self._model = None
        self._residual_scorer = TrimmedScorer(grace_period=10)
        self._residual_filter = quantile_filter(self._residual_scorer, quantile)
        self._prebuffer = []
        self._bucket_acc = []
        self.k = 1  # pushes per bucket
        self.L = None  # seasonal length in buckets
        self.bucket_size_s = None
        self.guard_tripped = False
        self.iat_at_resolution = None
        self.model_updates = 0

    @property
    def resolved(self) -> bool:
        return self._model is not None

    def reset_buffer(self):
        """Drop pre-resolution samples: the rate guard changed the scoring granularity."""
        self._prebuffer.clear()

    def resolve(self, median_iat: float):
        """Fix L from the measured cadence, apply the rollup guard, replay buffered values."""
        raw_length = max(2, round(self.period_s / median_iat))
        if self.bucket_mode == "auto":
            if raw_length > MAX_SEASONAL_LENGTH:
                self.k = math.ceil(raw_length / TARGET_SEASONAL_LENGTH)
                self.guard_tripped = True
        elif self.bucket_mode == "off":
            self.k = 1
        else:  # explicit duration in seconds
            self.k = max(1, round(self.bucket_mode / median_iat))
        self.L = max(2, round(raw_length / self.k))
        if self.k > 1:
            self.bucket_size_s = round(self.k * median_iat, 3)
        self.iat_at_resolution = median_iat

        if self.model_kind == "sarimax":
            self._model = time_series.SNARIMAX(p=1, d=0, q=1, m=self.L, sp=1, sq=1)
        else:
            self._model = time_series.HoltWinters(alpha=HW_ALPHA, beta=HW_BETA, gamma=HW_GAMMA, seasonality=self.L)

        buffered, self._prebuffer = self._prebuffer, []
        for value in buffered:
            self.update(value)

    def update(self, value: float) -> Evaluation | None:
        """Feed one raw push; returns an Evaluation once per completed bucket."""
        if not self.resolved:
            self._prebuffer.append(value)
            return None
        self._bucket_acc.append(value)
        if len(self._bucket_acc) < self.k:
            return None
        bucket_value = sum(self._bucket_acc) / len(self._bucket_acc)
        self._bucket_acc = []
        return self._score_and_learn(bucket_value)

    def _score_and_learn(self, y: float) -> Evaluation:
        predicted = self.forecast(1)
        predicted = predicted[0] if predicted else None
        if predicted is None:
            score, anomalous, residual, threshold = 0.0, False, None, None
        else:
            residual = y - predicted
            self._residual_scorer.observe(residual)
            score = self._residual_filter.score_one(None, residual)
            anomalous = self._residual_filter.classify(score)
            threshold = self._residual_filter.quantile.get()
            self._residual_filter.learn_one(None, residual)
        self._model.learn_one(y)
        self.model_updates += 1
        stats = {"residual": residual, "threshold": threshold}
        if self.bucket_size_s is not None:
            stats["bucket_size"] = self.bucket_size_s
        return Evaluation(score, anomalous, predicted, stats)

    def forecast(self, horizon: int) -> list[float] | None:
        """Model forecast, or None while the model cannot produce one yet."""
        if not self.resolved or horizon < 1:
            return None
        try:
            return self._model.forecast(horizon=horizon)
        except Exception:
            return None

    @property
    def detector(self):
        return self._model


class TrendForecaster:
    """Trend-only Holt-Winters used by forecast monitors without a declared period."""

    def __init__(self):
        self._model = time_series.HoltWinters(alpha=HW_ALPHA, beta=HW_BETA)

    def learn(self, value: float):
        self._model.learn_one(value)

    def forecast(self, horizon: int) -> list[float] | None:
        if horizon < 1:
            return None
        try:
            return self._model.forecast(horizon=horizon)
        except Exception:
            return None

    @property
    def detector(self):
        return self._model


class DriftPath:
    """Level-shift detection on the raw value stream, always on and O(1) per point."""

    def __init__(self, kind: str, ph_threshold: float, warmup: int):
        self.kind = kind
        self.updates = 0
        if kind == "adwin":
            self._detector = drift.ADWIN()
        else:
            self._detector = drift.PageHinkley(delta=0.005, threshold=ph_threshold, min_instances=warmup)
        self._mean = None  # slow EWMA, the "expected" level reported on drift events

    def update(self, value: float) -> bool:
        self.updates += 1
        self._detector.update(value)
        fired = self._detector.drift_detected
        if not fired:
            self._mean = value if self._mean is None else 0.05 * value + 0.95 * self._mean
        return fired

    @property
    def expected(self) -> float | None:
        return self._mean

    @property
    def detector(self):
        return self._detector


class ForecastRunner:
    """Projects the fitted model forward and fires once per predicted-breach episode."""

    def __init__(self, horizon_s: float):
        self.horizon_s = horizon_s
        self._breach_active = False

    def evaluate(self, forecast: list[float] | None, step_s: float, guard: LimitsGuard) -> dict | None:
        """Returns breach info {projected, eta_s} when a new breach episode starts."""
        breach = None
        for index, projected in enumerate(forecast or []):
            if guard.crossed(projected):
                breach = {"projected": projected, "eta_s": (index + 1) * step_s}
                break
        if breach is None:
            self._breach_active = False
            return None
        if self._breach_active:
            return None
        self._breach_active = True
        return breach

    def horizon_steps(self, step_s: float | None) -> int:
        if not step_s or step_s <= 0:
            return 0
        return max(1, round(self.horizon_s / step_s))
