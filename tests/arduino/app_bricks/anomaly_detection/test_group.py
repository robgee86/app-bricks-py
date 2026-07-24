# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import random

import pytest

from arduino.app_bricks.anomaly_detection import AnomalyDetectionGroup

T0 = 1_000_000.0


def feed_coupled(group, count, start=0, noise=0.01):
    """Two signals driven by one hidden load: a normal, coupled regime."""
    for index in range(start, start + count):
        load = random.uniform(0.2, 0.8)
        group.push(
            vibration=load * 0.3 + random.gauss(0, noise),
            power=100 + load * 40 + random.gauss(0, noise * 100),
            at=T0 + index,
        )
    return start + count


# ---- broadcast contract ------------------------------------------------------------


def test_scalar_broadcasts_and_array_configures_by_position(capsys):
    AnomalyDetectionGroup(
        "rig",
        ["vibration", "power"],
        period=[None, "1d"],
        sensitivity=["low", "medium"],
        persist=False,
    )
    echo = capsys.readouterr().out.strip()
    assert echo == "rig: vibration[sensitivity=low] power[period=1d, sensitivity=medium] · watch=joint · evaluate=on_any"


def test_wrong_array_length_raises_including_length_one():
    with pytest.raises(ValueError, match="period.*length 1.*2 signals"):
        AnomalyDetectionGroup("rig", ["vibration", "power"], period=["1d"], persist=False)
    with pytest.raises(ValueError, match="sensitivity.*length 3.*2 signals"):
        AnomalyDetectionGroup("rig", ["vibration", "power"], sensitivity=["low", "low", "low"], persist=False)


def test_flattened_limits_pair_gets_a_hint():
    with pytest.raises(ValueError, match=r"did you mean limits=\(0, 100\)\?"):
        AnomalyDetectionGroup("rig", ["vibration", "power"], limits=[0, 100], persist=False)
    # Mixed numeric slots are still caught, just without the reconstruction hint.
    with pytest.raises(ValueError, match=r"slots must be \(lo, hi\) tuples or None"):
        AnomalyDetectionGroup("rig", ["vibration", "power"], limits=[0, (0, 100)], persist=False)


def test_tuple_limits_broadcast_as_one_pair(recorder):
    group = AnomalyDetectionGroup("rig", ["a", "b"], limits=(0, 100), persist=False)
    recorder.attach(group)
    group.push(a=150.0, at=T0)
    group.push(b=150.0, at=T0 + 1)
    assert [event.metric for event in recorder.of_kind("anomaly")] == ["rig.a", "rig.b"]


def test_group_level_kwargs_never_accept_arrays():
    with pytest.raises(ValueError, match="watch.*group"):
        AnomalyDetectionGroup("rig", ["a", "b"], watch=["joint", "joint"], persist=False)
    with pytest.raises(ValueError, match="evaluate.*group"):
        AnomalyDetectionGroup("rig", ["a", "b"], evaluate=["on_any", "on_any"], persist=False)


def test_sensitivity_array_leaves_joint_score_at_medium():
    group = AnomalyDetectionGroup("rig", ["a", "b"], sensitivity=["low", "high"], persist=False)
    assert group.joint_sensitivity == "medium"
    group.joint_sensitivity = "high"  # pro override
    assert group.joint_sensitivity == "high"

    scalar = AnomalyDetectionGroup("rig2", ["a", "b"], sensitivity="low", persist=False)
    assert scalar.joint_sensitivity == "low"


# ---- construction validation -------------------------------------------------------


def test_member_ceiling_and_floor():
    with pytest.raises(ValueError, match="at most 5"):
        AnomalyDetectionGroup("big", ["a", "b", "c", "d", "e", "f"], persist=False)
    with pytest.raises(ValueError, match="at least 2"):
        AnomalyDetectionGroup("small", ["a"], persist=False)


def test_evaluate_validation():
    with pytest.raises(ValueError, match="does not match a declared signal"):
        AnomalyDetectionGroup("rig", ["a", "b"], evaluate="on_c", persist=False)
    with pytest.raises(ValueError, match="every:30s"):
        AnomalyDetectionGroup("rig", ["a", "b"], evaluate="every", persist=False)
    with pytest.raises(ValueError, match="watch"):
        AnomalyDetectionGroup("rig", ["a", "b"], watch="everything", persist=False)


def test_push_rejects_unknown_signals():
    group = AnomalyDetectionGroup("rig", ["a", "b"], persist=False)
    with pytest.raises(ValueError, match="unknown signal"):
        group.push(c=1.0)
    with pytest.raises(ValueError, match="at least one"):
        group.push()


# ---- per-member behavior -----------------------------------------------------------


def test_member_limits_fire_with_member_metric(recorder):
    group = AnomalyDetectionGroup("rig", ["vibration", "power"], limits=[None, (0, 110)], persist=False)
    recorder.attach(group)
    group.push(vibration=0.5, power=150.0, at=T0)
    breaches = recorder.of_kind("anomaly")
    assert len(breaches) == 1
    assert breaches[0].metric == "rig.power"
    assert breaches[0].stats.reason == "limit"


def test_evaluate_every_uses_the_timer():
    import time as time_module

    group = AnomalyDetectionGroup("rig", ["a", "b"], evaluate="every:1h", persist=False)
    now = time_module.time()
    for index in range(10):
        group.push(a=random.gauss(0, 1), b=random.gauss(0, 1), at=now - 10 + index)
    assert group.stats.evaluations == 0, "arrivals must not trigger evaluations in timer mode"
    group.loop()  # first timer tick evaluates immediately, sample-and-hold
    assert group.stats.evaluations == 1
    group.loop()  # within the same period: no new evaluation
    assert group.stats.evaluations == 1


def test_evaluate_on_signal_gates_evaluations():
    group = AnomalyDetectionGroup("rig", ["vibration", "power"], evaluate="on_power", persist=False)
    for index in range(10):
        group.push(vibration=0.5, at=T0 + index)
    assert group.stats.evaluations == 0
    group.push(power=100.0, at=T0 + 10)
    group.push(power=100.0, at=T0 + 11)
    assert group.stats.evaluations > 0


# ---- joint mode --------------------------------------------------------------------


def test_joint_anomaly_with_coincidence_enrichment(recorder):
    group = AnomalyDetectionGroup("rig", ["vibration", "power"], persist=False)
    recorder.attach(group)
    end = feed_coupled(group, 600)
    assert group.ready

    for index in range(end, end + 12):
        group.push(vibration=0.9, power=90.0, at=T0 + index)

    joint = [event for event in recorder.of_kind("anomaly") if event.metric == "rig"]
    assert joint, "an abnormal combination must fire a joint anomaly"
    event = joint[-1]
    assert set(event.value) == {"vibration", "power"}
    assert set(event.expected) == {"vibration", "power"}
    assert 0.0 <= event.score <= 1.0
    assert event.stats.coherent in (True, False)
    assert isinstance(event.stats.agreement, list)
    assert event.summary is not None


def test_joint_not_ready_before_volume():
    group = AnomalyDetectionGroup("rig", ["a", "b"], persist=False)
    for index in range(100):
        group.push(a=random.gauss(0, 1), b=random.gauss(0, 1), at=T0 + index)
    assert not group.ready


def test_group_warmup_override():
    group = AnomalyDetectionGroup("rig", ["a", "b"], persist=False)
    group.warmup = 5
    for index in range(6):
        group.push(a=float(index), b=float(index), at=T0 + index)
    assert group.ready


# ---- relationship mode -------------------------------------------------------------


def test_relationship_break_detected(recorder):
    group = AnomalyDetectionGroup("pair", ["x", "y"], watch="relationship", sensitivity="high", persist=False)
    recorder.attach(group)
    for index in range(200):
        x = random.gauss(0, 1)
        group.push(x=x, y=2 * x + random.gauss(0, 0.05), at=T0 + index)
    assert group.ready
    assert [e for e in recorder.of_kind("anomaly") if e.metric == "pair"] == []

    for index in range(200, 500):
        group.push(x=random.gauss(0, 1), y=random.gauss(0, 1), at=T0 + index)
        broken = [e for e in recorder.of_kind("anomaly") if e.metric == "pair"]
        if broken:
            break
    assert broken, "decoupling two correlated signals must fire a relationship anomaly"
    assert broken[0].stats.r is not None
    assert broken[0].expected == pytest.approx(broken[0].stats.expected_r, abs=1e-3)


def test_three_member_relationship_flags_deviating_member(recorder):
    group = AnomalyDetectionGroup("trio", ["a", "b", "c"], watch="relationship", sensitivity="high", persist=False)
    recorder.attach(group)
    for index in range(250):
        base = random.uniform(0, 1)
        group.push(a=base, b=base * 2, c=base * 3, at=T0 + index)
    assert group.ready

    fired = []
    for index in range(250, 600):
        base = random.uniform(0, 1)
        group.push(a=base, b=base * 2, c=random.uniform(0, 3), at=T0 + index)
        fired = [e for e in recorder.of_kind("anomaly") if e.metric == "trio"]
        if fired:
            break
    assert fired
    assert fired[0].stats.deviating == "c"


# ---- staleness ---------------------------------------------------------------------


def test_stale_member_skips_scoring_and_fires_once(recorder):
    group = AnomalyDetectionGroup("rig", ["fast", "slow"], persist=False)
    recorder.attach(group)
    for index in range(50):
        group.push(fast=random.gauss(0, 1), slow=random.gauss(0, 1), at=T0 + index)
    evaluations = group.stats.evaluations

    # "slow" goes silent while "fast" keeps pushing; staleness kicks in once the age
    # exceeds max(3 x member IAT, 2 x evaluation interval), a handful of evaluations.
    for index in range(50, 80):
        group.push(fast=random.gauss(0, 1), at=T0 + index)
    stalled = recorder.of_kind("stalled")
    assert len(stalled) == 1
    assert stalled[0].metric == "rig.slow"
    assert group.stats.evaluations <= evaluations + 3, "joint scoring must be skipped while a member is stale"

    # The member coming back closes the staleness episode.
    for index in range(80, 85):
        group.push(fast=random.gauss(0, 1), slow=random.gauss(0, 1), at=T0 + index)
    for index in range(85, 115):
        group.push(fast=random.gauss(0, 1), at=T0 + index)
    assert len(recorder.of_kind("stalled")) == 2


# ---- persistence -------------------------------------------------------------------


def test_group_state_restored_across_instances():
    group = AnomalyDetectionGroup("plant", ["a", "b"])
    for index in range(50):
        group.push(a=random.gauss(0, 1), b=random.gauss(0, 1), at=T0 + index)
    evaluations = group.stats.evaluations
    assert evaluations > 0
    group.stop()

    restored = AnomalyDetectionGroup("plant", ["a", "b"])
    assert restored.stats.evaluations == evaluations

    changed = AnomalyDetectionGroup("plant", ["a", "b"], sensitivity="high")
    assert changed.stats.evaluations == 0, "a different fingerprint must re-warm"
