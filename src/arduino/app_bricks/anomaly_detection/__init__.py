# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Self-calibrating anomaly detection for sensor streams.

Speak intent, not algorithms: AnomalyDetection watches one signal,
AnomalyDetectionGroup watches how 2-5 related signals behave together. Detectors,
thresholds and warm-up are resolved internally from a single sensitivity dial.
"""

from .events import AnomalyEvent, Stats
from .group import AnomalyDetectionGroup
from .monitor import AnomalyDetection

__all__ = ["AnomalyDetection", "AnomalyDetectionGroup", "AnomalyEvent", "Stats"]
