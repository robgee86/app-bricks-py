# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Watch a single simulated temperature signal with hard safety limits."""

import random

from arduino.app_bricks.anomaly_detection import AnomalyDetection
from arduino.app_utils import App

monitor = AnomalyDetection("temperature", limits=(0, 90))
monitor.on_ready(lambda e: print("temperature: learned what normal looks like"))
monitor.on_anomaly(lambda e: print(f"anomaly: {e.value:.1f} (expected ~{e.expected:.1f}, score {e.score:.2f})"))
monitor.on_normal(lambda e: print("back to normal"))
monitor.on_drift(lambda e: print(f"level shift: now around {e.value:.1f}"))


def read_sensor():
    monitor.push(random.gauss(23.0, 0.8))


App.run(user_loop=read_sensor)
