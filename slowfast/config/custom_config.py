#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Kernel size of gaze label
    _C.DATA.GAUSSIAN_KERNEL = 19

    # ---------------------------------------------------------------------------- #
    # Audio-visual gaze anticipation/estimation options
    # ---------------------------------------------------------------------------- #
    # If True, the dataloader return target frames for gaze forecasting
    _C.DATA_LOADER.RETURN_TARGET_FRAME = False

    # Path to the audio encoder checkpoint. If False, don't load pretrained parameters.
    _C.TRAIN.AUDIO_CHECKPOINT_FILE_PATH = ""

    # Weight of NCE loss for Audio-visual gaze modeling model
    _C.MODEL.LOSS_ALPHA = 1.0

    # If True, return spatial attention weights of audio signals from the spatial fusion module
    _C.MVIT.SPATIAL_AUDIO_ATTN = False
