# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import sys
import os
sys.path.append(os.getcwd())

from tool.batch.roi_layers import _C

nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
import torch
