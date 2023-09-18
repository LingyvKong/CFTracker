# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .nms.py_cpu_nms import py_cpu_nms

USE_GPU_NMS = False


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if USE_GPU_NMS and not force_cpu:
        from .nms.gpu_nms import gpu_nms
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return py_cpu_nms(dets, thresh)
