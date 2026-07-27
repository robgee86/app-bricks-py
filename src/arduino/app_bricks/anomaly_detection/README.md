# Anomaly Detection

Self-calibrating anomaly detection for numeric sensor streams. Push values, get callbacks:
no thresholds to pick, no models to train, no tuning beyond a single sensitivity dial.

## Overview

The Anomaly Detection brick provides two building blocks:

- `AnomalyDetection` watches a single signal and learns what normal looks like
- `AnomalyDetectionGroup` watches how 2-5 related signals behave together

Both fire plain callbacks with a flat event record (`metric`, `kind`, `value`, `expected`,
`score`, `at`, `stats`) and persist their learned state across restarts automatically.

## Features

- Adaptive detection with automatic warm-up, no configuration required
- Safe out of the box on spiky, glitchy, quantized and slowly drifting feeds
- Seasonal signals via `period="1d"`
- Hard safety `limits`, active from the very first push (no warm-up)
- Level-shift detection (`on_drift`) always on, for free
- Breach prediction with `forecast="30m"` (`on_forecast`)
- Stalled-feed detection (`on_stalled`)
- AnomalyDetectionGroup modes: `watch="joint"` (is this combination abnormal?) and
  `watch="relationship"` (has the usual coupling broken?), with built-in coincidence
  analysis to tell one problem from two

## Code example and usage

Watching one signal:

```python
from arduino.app_bricks.anomaly_detection import AnomalyDetection
from arduino.app_utils import App

m = AnomalyDetection("temperature", limits=(0, 90))
m.on_anomaly(lambda e: print(f"{e.metric}: {e.value} (expected ~{e.expected}, score {e.score:.2f})"))
m.on_normal(lambda e: print(f"{e.metric}: back to normal"))

def read_sensor():
    m.push(read_temperature())

App.run(user_loop=read_sensor)
```

A daily-seasonal signal with sensitivity tuning and breach prediction:

```python
m = AnomalyDetection("boiler", period="1d", limits=(None, 85), forecast="30m", sensitivity="low")
m.on_forecast(lambda e: print(f"predicted {e.expected:.1f} in ~{e.stats.eta_s:.0f}s"))
```

Watching signals together:

```python
from arduino.app_bricks.anomaly_detection import AnomalyDetectionGroup

g = AnomalyDetectionGroup("rig", ["vibration", "power"], sensitivity="high")
g.on_anomaly(lambda e: print(e.summary or f"{e.metric} anomaly"))
g.push(vibration=0.21, power=118.0)
```

Per-member configuration follows the broadcast contract: a scalar applies to every member,
a list configures members by position (`None` = that member's default) and must have exactly
one slot per signal:

```python
g = AnomalyDetectionGroup("rig", ["vibration", "power"], period=[None, "1d"], sensitivity=["low", "medium"])
```

## Events

| Callback      | Fires when                                                        |
| ------------- | ----------------------------------------------------------------- |
| `on_anomaly`  | A learned detection or hard-limit breach (`stats.reason="limit"`) |
| `on_normal`   | The signal recovers after an anomaly episode                      |
| `on_drift`    | The signal's level shifts to a new normal                         |
| `on_forecast` | A limit breach is predicted within the forecast horizon           |
| `on_stalled`  | The feed goes silent                                              |
| `on_ready`    | Warm-up completes (check `m.ready` anytime)                       |

A seasonal monitor calibrates its season length from the push cadence it measures during
warm-up. If the cadence later changes by more than 2x, `on_drift` fires with
`stats.reason="cadence_change"` and the seasonal model keeps running against the old
calibration: restart the app to re-calibrate (fresh cadence measurement).

Persistence is on by default (`persist=False` opts out), storing model state under
`./data/anomaly_detection/`; changing the configuration intentionally invalidates saved
state so a model is never scored against assumptions it was not fitted under.
