# Copyright (c) FelixFu. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmdlp.registry import HOOKS


@HOOKS.register_module()
class PruningChannelsHook(Hook):
    """
    使用剪枝算法对模型的通道进行剪枝, 以减少模型的计算量和参数量.
    参考: https://github.com/VainF/Torch-Pruning

    Args:
        
    """

    def __init__(self):
        pass