# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Intent resolution tables: sensitivity fan-out, durations, gating ladder, fingerprints."""

import hashlib
import re
from dataclasses import dataclass, fields, replace

SENSITIVITIES = ("low", "medium", "high")

# Ordered from most permissive to most conservative: lone-signal damping moves one step right.
GATE_LADDER = ((1, 1), (2, 3), (3, 5), (4, 7))

# Non-seasonal learned detectors warm up over this many points, scaled by the sensitivity multiplier.
BASE_WARMUP_POINTS = 120
# Drift detection warm-up is fixed, independent of sensitivity.
DRIFT_WARMUP_POINTS = 60
# AnomalyDetectionGroup joint (HalfSpaceTrees) needs volume: joint evaluations, scaled by the sensitivity multiplier.
GROUP_JOINT_WARMUP = 240
# AnomalyDetectionGroup relationship warm-up: aligned evaluations, scaled by the sensitivity multiplier.
GROUP_RELATIONSHIP_WARMUP = 120

# Seasonal lengths above this trip the rollup guard when bucket="auto".
MAX_SEASONAL_LENGTH = 5000
# The rollup guard aggregates pushes so the effective seasonal length stays at or below this.
TARGET_SEASONAL_LENGTH = 1440

# Points required before the inter-arrival time is considered measured and L can be resolved.
CADENCE_RESOLUTION_POINTS = 30

# A drift alarm flushes the learned window only if the shift sustains: a lone spike also
# trips PageHinkley, and recalibrating on a spike would adopt it as the new normal.
FLUSH_CONFIRM_POINTS = 10
FLUSH_CONFIRM_SIGMAS = 4.0

MAX_GROUP_MEMBERS = 5

AUTOSAVE_INTERVAL_S = 300.0


@dataclass(frozen=True)
class Sensitivity:
    """Internal tolerances resolved from the single user-facing sensitivity dial."""

    quantile: float  # QuantileFilter cutoff Q_sens
    gate: tuple[int, int]  # fire on X of last Y evaluations
    hysteresis: int  # consecutive normal points to emit on_normal
    warmup_mult: float  # M_sens applied to base warm-up
    ph_threshold: float  # PageHinkley threshold T_sens
    score_floor: float  # events below this score are not emitted


SENSITIVITY_TABLE = {
    "low": Sensitivity(quantile=0.999, gate=(3, 5), hysteresis=5, warmup_mult=1.5, ph_threshold=50.0, score_floor=0.90),
    "medium": Sensitivity(quantile=0.995, gate=(2, 3), hysteresis=3, warmup_mult=1.0, ph_threshold=30.0, score_floor=0.80),
    "high": Sensitivity(quantile=0.98, gate=(1, 1), hysteresis=2, warmup_mult=0.75, ph_threshold=15.0, score_floor=0.65),
}

_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$")


def parse_duration(value: str, kwarg: str) -> float:
    """Parse a duration string like "30s", "5m", "1h", "1d" into seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError(f"{kwarg}={value!r}: duration must be positive")
        return float(value)
    match = _DURATION_RE.match(value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f'{kwarg}={value!r}: expected a duration like "30s", "5m", "1h" or "1d"')
    seconds = float(match.group(1)) * _DURATION_UNITS[match.group(2)]
    if seconds <= 0:
        raise ValueError(f"{kwarg}={value!r}: duration must be positive")
    return seconds


def resolve_sensitivity(value, kwarg: str = "sensitivity") -> Sensitivity:
    """Resolve a sensitivity name, or a pro dict of internal tolerances.

    A dict overrides individual internals (quantile, gate, hysteresis, warmup_mult,
    ph_threshold, score_floor); unspecified keys keep "medium"'s values.
    """
    if isinstance(value, dict):
        return _resolve_sensitivity_overrides(value, kwarg)
    if value not in SENSITIVITY_TABLE:
        raise ValueError(f"{kwarg}={value!r}: must be one of {', '.join(SENSITIVITIES)}, or a dict of internal tolerances")
    return SENSITIVITY_TABLE[value]


def _resolve_sensitivity_overrides(overrides: dict, kwarg: str) -> Sensitivity:
    known = {field.name for field in fields(Sensitivity)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"{kwarg}: unknown key(s) {', '.join(sorted(unknown))}; valid keys: {', '.join(sorted(known))}")
    resolved = replace(SENSITIVITY_TABLE["medium"], **overrides)
    if not 0.5 < resolved.quantile < 1.0:
        raise ValueError(f"{kwarg}: quantile must be between 0.5 and 1.0 (exclusive)")
    gate = resolved.gate
    if not (isinstance(gate, tuple) and len(gate) == 2 and all(isinstance(g, int) for g in gate) and 1 <= gate[0] <= gate[1]):
        raise ValueError(f"{kwarg}: gate must be an (x, y) tuple of ints with 1 <= x <= y (fire on x of the last y evaluations)")
    if not (isinstance(resolved.hysteresis, int) and resolved.hysteresis >= 1):
        raise ValueError(f"{kwarg}: hysteresis must be an int >= 1")
    if resolved.warmup_mult <= 0 or resolved.ph_threshold <= 0:
        raise ValueError(f"{kwarg}: warmup_mult and ph_threshold must be positive")
    if not 0.0 <= resolved.score_floor < 1.0:
        raise ValueError(f"{kwarg}: score_floor must be in [0.0, 1.0)")
    return resolved


def raise_gate(gate: tuple[int, int]) -> tuple[int, int]:
    """Return the gate one step more conservative on the ladder (lone-signal damping)."""
    try:
        index = GATE_LADDER.index(gate)
    except ValueError:
        return gate
    return GATE_LADDER[min(index + 1, len(GATE_LADDER) - 1)]


def validate_limits(limits, kwarg: str = "limits") -> tuple[float | None, float | None]:
    """Validate a (lo, hi) hard-bounds pair; one side may be None."""
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError(f"{kwarg}={limits!r}: expected a (lo, hi) tuple; one side may be None")
    lo, hi = limits
    for bound in (lo, hi):
        if bound is not None and not isinstance(bound, (int, float)):
            raise ValueError(f"{kwarg}={limits!r}: bounds must be numbers or None")
    if lo is None and hi is None:
        raise ValueError(f"{kwarg}={limits!r}: at least one bound must be set")
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"{kwarg}={limits!r}: lo must be smaller than hi")
    return (lo, hi)


def broadcast_member_kwarg(kwarg: str, value, signals: list[str], default) -> list:
    """Resolve a member-level AnomalyDetectionGroup kwarg per the broadcast contract.

    Scalars broadcast to every member; a list configures members by position, with None
    slots meaning that member's default. Anything but an exact-length list raises.
    """
    if isinstance(value, list):
        if len(value) != len(signals):
            raise ValueError(
                f"{kwarg}: got a list of length {len(value)} but the group has {len(signals)} signals; "
                f"per-member arrays must have exactly one slot per signal (scalars already broadcast)"
            )
        return [default if slot is None else slot for slot in value]
    return [default if value is None else value for _ in signals]


def reject_group_kwarg_array(kwarg: str, value):
    """AnomalyDetectionGroup-level kwargs describe the group, not its members: arrays are always an error."""
    if isinstance(value, list):
        raise ValueError(f"{kwarg}: describes the group as a whole, not its members, so it never accepts an array")


def reject_flattened_limits(kwarg: str, value):
    """Catch limits=[0, 100]: a length-valid array whose slots are numbers, not limit specs."""
    if not isinstance(value, list):
        return
    numeric = [slot for slot in value if isinstance(slot, (int, float)) and not isinstance(slot, bool)]
    if not numeric:
        return
    hint = f" — did you mean {kwarg}=({value[0]}, {value[1]})?" if len(value) == 2 and len(numeric) == 2 else ""
    raise ValueError(f"{kwarg}={value!r}: array slots must be (lo, hi) tuples or None{hint}")


def config_fingerprint(resolved: dict) -> str:
    """Hash the resolved internals (not the intent strings) into a persistence key.

    Any change to the resolved configuration invalidates persisted state so a model is
    never scored against assumptions it was not fitted under.
    """
    canonical = repr(sorted(resolved.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
