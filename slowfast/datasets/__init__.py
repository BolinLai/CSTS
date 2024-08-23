#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa

# Remember to add new dataset here. @DATASET_REGISTRY.register() is not executed otherwise.
from .ego4d_avgaze import Ego4d_av_gaze
from .ego4d_avgaze_forecast import Ego4d_av_gaze_forecast
from .aria_avgaze import Aria_av_gaze
from .aria_avgaze_forecast import Aria_av_gaze_forecast

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
