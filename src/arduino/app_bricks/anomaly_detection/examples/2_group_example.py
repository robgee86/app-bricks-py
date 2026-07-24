# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Watch two related simulated signals together: is this combination abnormal?"""

import random

from arduino.app_bricks.anomaly_detection import AnomalyDetectionGroup
from arduino.app_utils import App

group = AnomalyDetectionGroup("rig", ["vibration", "power"], sensitivity="high")
group.on_ready(lambda e: print("rig: learned what normal looks like"))
group.on_anomaly(lambda e: print(e.summary or f"{e.metric}: anomaly (score {e.score:.2f})"))
group.on_stalled(lambda e: print(f"{e.metric}: feed went silent"))


def read_sensors():
    load = random.uniform(0.2, 0.8)
    group.push(vibration=load * 0.3 + random.gauss(0, 0.01), power=100 + load * 40 + random.gauss(0, 1))


App.run(user_loop=read_sensors)
