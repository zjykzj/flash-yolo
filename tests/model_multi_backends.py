# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/23 20:51
@File    : model_multi_backends.py
@Author  : zj
@Description: 
"""

from yolov5.nn.backend import DetectMultiBackend

model = DetectMultiBackend("yolov5s_c.pt")
