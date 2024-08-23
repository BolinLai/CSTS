#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Kernel size of gaze label
    _C.DATA.GAUSSIAN_KERNEL = 19

    # ---------------------------------------------------------------------------- #
    # Global-local correlation options
    # ---------------------------------------------------------------------------- #
    # If True, use global embed in the network
    _C.MVIT.GLOBAL_EMBED_ON = False

    # Drop path rate of GLC module
    _C.MVIT.GLC_DROP_PATH = 0.0

    # ---------------------------------------------------------------------------- #
    # Audio-visual gaze anticipation/estimation options
    # ---------------------------------------------------------------------------- #
    # If True, the dataloader return target frames for gaze forecasting
    _C.DATA_LOADER.RETURN_TARGET_FRAME = False

    # Path to the audio encoder checkpoint. If False, don't load pretrained parameters.
    _C.TRAIN.AUDIO_CHECKPOINT_FILE_PATH = ""

    # Weight of NCE loss for Audio-visual gaze modeling model
    _C.MODEL.LOSS_ALPHA = 1.0

    # Weight of negative samples in cross-entropy loss for gaze shift prediction
    _C.MODEL.LOSS_WEIGHT_OF_GS_NEGATIVE_SAMPLE = 1.0

    # If True, return spatial attention weights of audio signals from the spatial fusion module
    _C.MVIT.SPATIAL_AUDIO_ATTN = False

    # ---------------------------------------------------------------------------- #
    # MotionFormer options
    # ---------------------------------------------------------------------------- #
    _C.VIT = CfgNode()

    # Patch-size spatial to tokenize input
    _C.VIT.PATCH_SIZE = 16

    # Patch-size temporal to tokenize input
    _C.VIT.PATCH_SIZE_TEMP = 2

    # Number of input channels
    _C.VIT.CHANNELS = 3

    # Embedding dimension
    _C.VIT.EMBED_DIM = 768

    # Depth of transformer: number of layers
    _C.VIT.DEPTH = 12

    # number of attention heads
    _C.VIT.NUM_HEADS = 12

    # expansion ratio for MLP
    _C.VIT.MLP_RATIO = 4

    # add bias to QKV projection layer
    _C.VIT.QKV_BIAS = True

    # video input
    _C.VIT.VIDEO_INPUT = True

    # temporal resolution i.e. number of frames
    _C.VIT.TEMPORAL_RESOLUTION = 8

    # use MLP classification head
    _C.VIT.USE_MLP = False

    # Dropout rate for
    _C.VIT.DROP = 0.0

    # Stochastic drop rate
    _C.VIT.DROP_PATH = 0.0

    # Dropout rate for MLP head
    _C.VIT.HEAD_DROPOUT = 0.0

    # Dropout rate for positional embeddings
    _C.VIT.POS_DROPOUT = 0.0

    # Dropout rate
    _C.VIT.ATTN_DROPOUT = 0.0

    # Activation for head
    _C.VIT.HEAD_ACT = "tanh"

    # Use IM pretrained weights
    _C.VIT.IM_PRETRAINED = True

    # Pretrained weights type
    _C.VIT.PRETRAINED_WEIGHTS = "vit_1k"

    # Type of position embedding
    _C.VIT.POS_EMBED = "separate"

    # Self-Attention layer
    _C.VIT.ATTN_LAYER = "trajectory"

    # Flag to use original trajectory attn code
    _C.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True

    # Approximation type
    _C.VIT.APPROX_ATTN_TYPE = "none"

    # Approximation Dimension
    _C.VIT.APPROX_ATTN_DIM = 128

    # If True, use class embed token in the network
    _C.VIT.CLS_EMBED_ON = True

    # If True, use global embed in the network
    _C.VIT.GLOBAL_EMBED_ON = False

    # Drop path rate of GLC module
    _C.VIT.GLC_DROP_PATH = 0.0

    pass
