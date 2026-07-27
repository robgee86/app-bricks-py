# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import random
import time

import pytest

from arduino.app_bricks.anomaly_detection import AnomalyDetection

T0 = 1_000_000.0


def feed_gaussian(monitor, count, mean=50.0, sigma=1.0, start=0, step=1.0):
    for index in range(start, start + count):
        monitor.push(mean + random.gauss(0, sigma), at=T0 + index * step)
    return start + count


# ---- construction ------------------------------------------------------------------


def test_invalid_combinations_raise():
    with pytest.raises(ValueError, match="forecast"):
        AnomalyDetection("m1", forecast="30m", persist=False)
    with pytest.raises(ValueError, match="bucket"):
        AnomalyDetection("m2", bucket="30s", persist=False)
    with pytest.raises(ValueError, match="sarimax"):
        AnomalyDetection("m4", model="sarimax", persist=False)
    with pytest.raises(ValueError, match="sensitivity"):
        AnomalyDetection("m5", sensitivity="extreme", persist=False)
    with pytest.raises(ValueError, match="duration"):
        AnomalyDetection("m6", period="tomorrow", persist=False)
    with pytest.raises(ValueError, match="limits"):
        AnomalyDetection("m7", limits=(10, 5), persist=False)
    with pytest.raises(ValueError, match="name"):
        AnomalyDetection("", persist=False)


def test_construction_echo_prints_resolved_config(capsys):
    AnomalyDetection("temperature", limits=(0, 90), persist=False)
    assert capsys.readouterr().out.strip() == "temperature: limits=(0, 90), sensitivity=medium"


# ---- limits layer ------------------------------------------------------------------


def test_limits_fire_from_first_push_and_recover(recorder):
    monitor = AnomalyDetection("t", limits=(0, 90), persist=False)
    recorder.attach(monitor)

    monitor.push(120.0, at=T0)
    breach = recorder.of_kind("anomaly")
    assert len(breach) == 1
    assert breach[0].stats.reason == "limit"
    assert breach[0].expected == 90
    assert breach[0].score == 1.0
    assert breach[0].value == 120.0

    # Still out of bounds: no duplicate event.
    monitor.push(130.0, at=T0 + 1)
    assert len(recorder.of_kind("anomaly")) == 1

    # Recovery hysteresis (medium: 3 consecutive normal points), even before warm-up.
    for index in range(2, 5):
        monitor.push(50.0, at=T0 + index)
    assert recorder.kinds().count("normal") == 1


def test_one_sided_limits():
    monitor = AnomalyDetection("t", limits=(None, 90), persist=False)
    events = []
    monitor.on_anomaly(events.append)
    monitor.push(-1000.0, at=T0)
    assert events == []
    monitor.push(91.0, at=T0 + 1)
    assert len(events) == 1


# ---- warm-up and the learned detector ----------------------------------------------


def test_warmup_suppresses_learned_events_but_not_limits(recorder):
    monitor = AnomalyDetection("t", limits=(0, 90), persist=False)
    recorder.attach(monitor)
    feed_gaussian(monitor, 50)
    monitor.push(70.0, at=T0 + 51)  # extreme for N(50, 1) but within limits
    assert recorder.of_kind("anomaly") == []
    assert not monitor.ready


def test_ready_after_derived_warmup(recorder):
    monitor = AnomalyDetection("t", persist=False)
    recorder.attach(monitor)
    feed_gaussian(monitor, 119)
    assert not monitor.ready
    feed_gaussian(monitor, 1, start=119)
    assert monitor.ready
    assert recorder.kinds().count("ready") == 1


def test_sensitivity_scales_warmup():
    high = AnomalyDetection("h", sensitivity="high", persist=False)
    feed_gaussian(high, 90)
    assert high.ready
    low = AnomalyDetection("l", sensitivity="low", persist=False)
    feed_gaussian(low, 179)
    assert not low.ready


def test_warmup_override_replaces_formula():
    monitor = AnomalyDetection("t", persist=False)
    monitor.warmup = 10
    feed_gaussian(monitor, 10)
    assert monitor.ready


def test_gaussian_anomaly_and_recovery(recorder):
    monitor = AnomalyDetection("t", persist=False)
    recorder.attach(monitor)
    end = feed_gaussian(monitor, 200)

    # Medium gating fires on 2 of the last 3 evaluations.
    monitor.push(65.0, at=T0 + end)
    monitor.push(65.0, at=T0 + end + 1)
    anomalies = recorder.of_kind("anomaly")
    assert len(anomalies) == 1
    event = anomalies[0]
    assert event.metric == "t"
    assert event.value == 65.0
    assert 45 < event.expected < 55
    assert event.score > 0.9
    assert abs(event.stats.z) > 5
    assert event.stats.detector is not None

    for index in range(3):
        monitor.push(50.0, at=T0 + end + 2 + index)
    assert recorder.kinds().count("normal") == 1


def test_high_sensitivity_fires_on_single_spike_low_does_not():
    high = AnomalyDetection("h", sensitivity="high", persist=False)
    fired = []
    high.on_anomaly(fired.append)
    end = feed_gaussian(high, 200)
    high.push(65.0, at=T0 + end)
    # 1-of-1 gating: the single spike fires immediately (high accepts noise, so other
    # borderline points may have fired too).
    assert any(event.value == 65.0 for event in fired)

    low = AnomalyDetection("l", sensitivity="low", persist=False)
    low_fired = []
    low.on_anomaly(low_fired.append)
    end = feed_gaussian(low, 250)
    low.push(65.0, at=T0 + end)
    low.push(50.0, at=T0 + end + 1)
    assert low_fired == []  # 3 of 5 not satisfied by one spike


def test_glitchy_feed_does_not_inflate_the_band():
    monitor = AnomalyDetection("spiky", persist=False)
    fired = []
    monitor.on_anomaly(fired.append)
    # 3% huge glitches during training must not widen what counts as normal.
    end = 0
    for index in range(300):
        glitch = random.choice([-20.0, 20.0]) if random.random() < 0.03 else 0.0
        monitor.push(50 + random.gauss(0, 1) + glitch, at=T0 + index)
        end = index + 1
    # Once the glitch burst leaves the rolling cutoff window, a 6-sigma point
    # still stands out: the trimmed band did not absorb the glitches.
    end = feed_gaussian(monitor, 500, start=end)
    monitor.push(56.0, at=T0 + end)
    monitor.push(56.0, at=T0 + end + 1)
    spikes = [event for event in fired if event.value == 56.0]
    assert spikes
    assert 48 < spikes[0].expected < 52


def test_quantized_feed_stays_quiet_but_catches_real_jumps(recorder):
    monitor = AnomalyDetection("quantized", persist=False)
    recorder.attach(monitor)
    # A healthy integer-step sensor: +/- one quantization step is normal jitter.
    for index in range(1500):
        monitor.push(float(round(random.gauss(22, 0.3))), at=T0 + index)
    assert recorder.of_kind("anomaly") == []
    for index in range(1500, 1503):
        monitor.push(28.0, at=T0 + index)
    assert any(event.value == 28.0 for event in recorder.of_kind("anomaly"))


def test_slow_drift_is_tracked_not_alarmed(recorder):
    monitor = AnomalyDetection("drifting", persist=False)
    recorder.attach(monitor)
    # Mean ramps 50 -> 60 over 3000 points, then holds: legitimate drift, not an anomaly.
    for index in range(4000):
        mean = 50 + min(10.0, index * 10.0 / 3000)
        monitor.push(mean + random.gauss(0, 1), at=T0 + index)
    assert len(recorder.of_kind("anomaly")) <= 2
    # The band followed the drift: a spike relative to the NEW level fires with expected ~60.
    monitor.push(66.0, at=T0 + 4000)
    monitor.push(66.0, at=T0 + 4001)
    spikes = [event for event in recorder.of_kind("anomaly") if event.value == 66.0]
    assert spikes
    assert 58 < spikes[0].expected < 62


# ---- high-rate feeds (rate guard) ----------------------------------------------------


def test_rate_guard_buckets_high_frequency_feeds():
    monitor = AnomalyDetection("gyro", persist=False)
    hz = 62.5
    for index in range(int(5 * hz)):  # 5 seconds of data
        monitor.push(random.gauss(0, 0.02), at=T0 + index / hz)
    assert monitor.stats.bucket_size == pytest.approx(1.0, rel=0.05)
    assert not monitor.ready, "warm-up must count buckets (seconds), not raw samples"


def test_normal_rate_feeds_stay_unbucketed():
    monitor = AnomalyDetection("slow", persist=False)
    feed_gaussian(monitor, 60)  # 1s cadence
    assert monitor.stats.bucket_size is None


def test_gyro_handling_is_normal_but_sustained_vibration_fires(recorder):
    import math

    hz, t = 10.0, 0.0
    monitor = AnomalyDetection("gyro_x", persist=False)
    recorder.attach(monitor)

    def push_seconds(seconds, wobble=0.0, freq=1.5):
        nonlocal t
        for _ in range(int(seconds * hz)):
            value = random.gauss(0, 0.02) + wobble * math.sin(2 * math.pi * freq * t)
            monitor.push(value, at=T0 + t)
            t += 1.0 / hz

    # Minutes of "rest with occasional ordinary handling": this is normal life.
    for _ in range(20):
        push_seconds(10)
        push_seconds(2, wobble=1.0)
    assert monitor.ready
    baseline = len(recorder.of_kind("anomaly"))
    for _ in range(5):
        push_seconds(10)
        push_seconds(2, wobble=1.0)
    assert len(recorder.of_kind("anomaly")) <= baseline + 1, "ordinary handling must not be chatty"

    # A genuinely sustained, unusually strong vibration: amplitude fault, level ~0.
    push_seconds(15, wobble=4.0, freq=2.0)
    fired = recorder.of_kind("anomaly")[baseline:]
    assert fired, "sustained abnormal vibration must fire"
    assert fired[-1].stats.lane == "spread", "an amplitude fault is a spread anomaly"
    assert fired[-1].stats.bucket_size == pytest.approx(1.0, rel=0.05)


# ---- drift -------------------------------------------------------------------------


def test_level_shift_emits_drift(recorder):
    monitor = AnomalyDetection("t", persist=False)
    recorder.attach(monitor)
    end = feed_gaussian(monitor, 150, sigma=0.1)
    for index in range(30):
        monitor.push(80.0 + random.gauss(0, 0.1), at=T0 + end + index)
    drifts = recorder.of_kind("drift")
    assert drifts, "PageHinkley should fire on a 300-sigma level shift"
    assert 45 < drifts[0].expected < 55  # the pre-shift level


def test_level_shift_recenters_the_band(recorder):
    monitor = AnomalyDetection("t", persist=False)
    recorder.attach(monitor)
    end = feed_gaussian(monitor, 200, sigma=0.1)
    # Step to a new level: anomaly, then drift confirmation flushes the scorer, then
    # the recalibrated band closes the episode around the new normal.
    for index in range(60):
        monitor.push(80.0 + random.gauss(0, 0.1), at=T0 + end + index)
    kinds = recorder.kinds()
    assert "drift" in kinds
    assert kinds.index("drift") < len(kinds) - 1 and "normal" in kinds[kinds.index("drift") :]
    # A spike relative to the NEW level fires with expected ~80, proving re-centering.
    monitor.push(81.0, at=T0 + end + 60)
    monitor.push(81.0, at=T0 + end + 61)
    spikes = [event for event in recorder.of_kind("anomaly") if event.value == 81.0]
    assert spikes
    assert 79 < spikes[0].expected < 81


# ---- seasonal path -----------------------------------------------------------------


def seasonal_value(index, period=20):
    import math

    return 10.0 + 5.0 * math.sin(2 * math.pi * index / period) + random.gauss(0, 0.2)


def test_seasonal_break_detected(recorder):
    monitor = AnomalyDetection("s", period="20s", persist=False)
    recorder.attach(monitor)
    for index in range(300):
        monitor.push(seasonal_value(index), at=T0 + index)
    assert monitor.ready
    # HW is crude right after the 2L warm-up; once settled, the season must be quiet.
    settled = [event for event in recorder.of_kind("anomaly") if event.at - T0 > 150]
    assert settled == []

    for index in range(300, 305):
        monitor.push(-10.0, at=T0 + index)
    breaks = [event for event in recorder.of_kind("anomaly") if event.value == -10.0]
    assert breaks, "a broken season must fire an anomaly"
    assert breaks[0].stats.residual is not None


def test_rollup_guard_buckets_long_seasons(monkeypatch):
    from arduino.app_bricks.anomaly_detection import events

    warnings = []
    monkeypatch.setattr(events.logger, "warning", warnings.append)
    monitor = AnomalyDetection("b", period="4h", persist=False)  # 14400 points at 1s cadence
    for index in range(40):
        monitor.push(10.0, at=T0 + index)
    assert monitor.stats.bucket_size == pytest.approx(10.0, rel=0.1)  # ceil(14400/1440) = 10s buckets
    assert any("averages" in message for message in warnings), "the rollup guard must warn about its consequence"


def test_explicit_bucket_duration():
    monitor = AnomalyDetection("b", period="10m", bucket="30s", persist=False)
    for index in range(40):
        monitor.push(10.0, at=T0 + index)
    assert monitor.stats.bucket_size == pytest.approx(30.0, rel=0.1)


def test_cadence_change_emits_drift(recorder):
    monitor = AnomalyDetection("s", period="20s", persist=False)
    recorder.attach(monitor)
    for index in range(60):
        monitor.push(seasonal_value(index), at=T0 + index)
    # The cadence collapses from 1s to 10s.
    for index in range(60):
        monitor.push(seasonal_value(index), at=T0 + 60 + index * 10.0)
    reasons = [event.stats.reason for event in recorder.of_kind("drift")]
    assert "cadence_change" in reasons


# ---- forecast ----------------------------------------------------------------------


def test_forecast_fires_before_breach(recorder):
    monitor = AnomalyDetection("f", limits=(None, 100), forecast="10s", persist=False)
    recorder.attach(monitor)
    monitor.warmup = 20
    value = 0.0
    for index in range(100):
        value = float(index)
        monitor.push(value, at=T0 + index)
        if recorder.of_kind("forecast"):
            break
    forecasts = recorder.of_kind("forecast")
    assert forecasts, "a steady ramp toward the limit must trigger on_forecast"
    event = forecasts[0]
    assert value < 100.0, "prediction must precede the actual breach"
    assert event.expected > 100.0
    assert event.stats.eta_s > 0


# ---- stalled -----------------------------------------------------------------------


def test_stalled_feed_detected(recorder):
    monitor = AnomalyDetection("t", persist=False)
    recorder.attach(monitor)
    now = time.time()
    for index in range(20):
        monitor.push(50.0, at=now - 120 + index)  # 1s cadence that stopped 100s ago
    monitor.loop()
    assert recorder.kinds().count("stalled") == 1
    monitor.loop()
    assert recorder.kinds().count("stalled") == 1  # once per staleness episode


# ---- persistence -------------------------------------------------------------------


def test_state_restored_across_instances(isolated_state_dir):
    monitor = AnomalyDetection("keep")
    feed_gaussian(monitor, 150)
    assert monitor.ready
    monitor.stop()
    assert list(isolated_state_dir.iterdir())

    restored = AnomalyDetection("keep")
    assert restored.ready, "restored state should skip re-warming"


def test_config_change_invalidates_state():
    monitor = AnomalyDetection("keep")
    feed_gaussian(monitor, 150)
    monitor.stop()

    changed = AnomalyDetection("keep", sensitivity="high")
    assert not changed.ready, "a different fingerprint must re-warm"


def test_persist_false_opts_out(isolated_state_dir):
    monitor = AnomalyDetection("ephemeral", persist=False)
    feed_gaussian(monitor, 10)
    monitor.stop()
    assert not isolated_state_dir.exists() or not list(isolated_state_dir.iterdir())


# ---- pro escape hatches ------------------------------------------------------------


def test_bring_your_own_detector_warns_on_ignored_kwargs(monkeypatch):
    from river import anomaly

    from arduino.app_bricks.anomaly_detection import events

    warnings = []
    monkeypatch.setattr(events.logger, "warning", warnings.append)
    monitor = AnomalyDetection("byo", detector=anomaly.GaussianScorer(grace_period=10), period="1d", persist=False)
    assert any("ignoring period" in message for message in warnings)
    feed_gaussian(monitor, 50)  # pushes flow through the replaced detector


def test_stats_surface():
    monitor = AnomalyDetection("t", limits=(0, 90), persist=False)
    monitor.push(50.0, at=T0)
    stats = monitor.stats
    assert stats.ready is False
    assert stats.points == 1
    assert stats.limits == (0, 90)
    assert stats.detector is not None


def test_cutoff_recovers_after_anomaly_bursts():
    monitor = AnomalyDetection("x", persist=False)
    fired = []
    monitor.on_anomaly(fired.append)
    end = feed_gaussian(monitor, 300)
    # Heavy anomaly cycles push the rolling cutoff toward 1.0...
    for _ in range(5):
        for _ in range(3):
            monitor.push(65.0, at=T0 + end)
            end += 1
        end = feed_gaussian(monitor, 20, start=end)
    # ...but it must recover once the bursts leave the cutoff window: a moderate
    # 3.2-sigma anomaly fires again after enough normal traffic.
    end = feed_gaussian(monitor, 600, start=end)
    before = len(fired)
    monitor.push(53.4, at=T0 + end)
    monitor.push(53.4, at=T0 + end + 1)
    assert len(fired) > before, "a past anomaly burst must not permanently raise the bar"


def test_sensitivity_accepts_tolerance_dict():
    # gate (1,1): a single spike fires without needing 2-of-3; other keys keep medium.
    monitor = AnomalyDetection("x", sensitivity={"gate": (1, 1), "score_floor": 0.7}, persist=False)
    fired = []
    monitor.on_anomaly(fired.append)
    end = feed_gaussian(monitor, 150)  # medium warm-up: ready after 120
    assert monitor.ready
    monitor.push(65.0, at=T0 + end)
    assert len(fired) == 1


def test_sensitivity_dict_validation():
    with pytest.raises(ValueError, match="unknown key.*quantille"):
        AnomalyDetection("x", sensitivity={"quantille": 0.99}, persist=False)
    with pytest.raises(ValueError, match="gate"):
        AnomalyDetection("x", sensitivity={"gate": (3, 1)}, persist=False)
    with pytest.raises(ValueError, match="quantile"):
        AnomalyDetection("x", sensitivity={"quantile": 1.5}, persist=False)
    with pytest.raises(ValueError, match="score_floor"):
        AnomalyDetection("x", sensitivity={"score_floor": 1.0}, persist=False)


def test_adwin_drift_swap():
    monitor = AnomalyDetection("t", drift="adwin", persist=False)
    fired = []
    monitor.on_drift(fired.append)
    for index in range(100):
        monitor.push(10.0 + random.gauss(0, 0.1), at=T0 + index)
    for index in range(100, 200):
        monitor.push(40.0 + random.gauss(0, 0.1), at=T0 + index)
    assert fired, "ADWIN should fire on a clear level shift"


def test_sarimax_seasonal_model():
    monitor = AnomalyDetection("s", period="20s", model="sarimax", persist=False)
    for index in range(100):
        monitor.push(seasonal_value(index), at=T0 + index)
    assert monitor.ready
    assert type(monitor.stats.detector).__name__ == "SNARIMAX"
