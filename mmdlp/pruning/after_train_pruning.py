# Copyright (c) FelixFu. All rights reserved.
import os.path as osp
import warnings
import time
from typing import Optional, Sequence, Union
import copy
from functools import partial
from thop import profile
import torch
import torch_pruning as tp

import mmengine
from mmengine.hooks import Hook
from mmdlp.registry import HOOKS
from mmengine.logging import print_log
from mmengine.device import get_device
from mmengine.dist import master_only, is_main_process, is_distributed
from mmengine.fileio import FileClient, join_path
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.runner import weights_to_cpu

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class AfterTrainPruning(Hook):
    """
    使用剪枝算法对模型的通道进行剪枝, 以减少模型的计算量和参数量.
    参考: https://github.com/VainF/Torch-Pruning

    Args:
        自定义的参数
        pruning_rate (float): the ratio of channels to prune. Default: 0.5.
        pruning_algorithm (str): the pruning algorithm to use. Default: taylor.
        ignore_layers_name (list): the names of layers to ignore. Default: None.
        global_pruning (bool): whether to prune the whole model. Default: True.
        isomorphic (bool): whether to prune the model isomorphically. Default: False.
        num_classes (int): the number of classes. Default: 80. 某些pruner需要, 如obdc. 暂不支持
        reg (float): the regularization factor. Default: 0.01. 某些pruner需要, 如slim. 暂不支持
        sparsity_learning (bool): whether to use sparsity learning. Default: False. 暂不支持
        delta_reg (float): the delta regularization factor. Default: 0.01. 某些pruner需要, 如growing_reg.   暂不支持
        
        torch_pruning库中的参数, 下面的参数是MagnitudePruner的参数, 详细的参数说明可以参考torch_pruning的文档
        **kwargs: other keyword arguments for torch_pruning.pruner.MagnitudePruner
    """
    priority = 'LOWEST'


    def __init__(self, 
                pruning_ratio: float = 0.5, 
                ignore_layers_name: list = None, 
                pruning_algorithm: str = "taylor", 
                global_pruning: bool = True,
                isomorphic: bool = False,
                num_classes: int = 80,
                reg: float = 0.01,
                sparsity_learning: bool = False,
                delta_reg: float = 0.01,
                **kwargs):
        self.pruning_ratio = pruning_ratio
        self.ignore_layers_name = ignore_layers_name
        self.pruning_algorithm = pruning_algorithm
        self.global_pruning = global_pruning
        self.isomorphic = isomorphic
        self.num_classes = num_classes
        self.reg = reg
        self.sparsity_learning = sparsity_learning
        self.delta_reg = delta_reg
        self.kwargs = kwargs

        assert self.pruning_ratio > 0 and self.pruning_ratio < 1, "pruning_ratio should be in (0, 1)"
        assert self.pruning_algorithm in ["random", "l1", "l2", "fpgm", "lamp", "group_norm",
                                          "taylor"], "Unsupported pruning algorithm"
        # 下面的方法中有些在train中有修改, 所以这里不再检查
        # assert self.pruning_algorithm in ["random", "l1", "l2", "fpgm", 
        #                                   "obdc", "lamp", "slim", "group_slim", 
        #                                   "group_norm", "group_sl", "growing_reg",
        #                                   "taylor"], "Unsupported pruning algorithm"

    def after_train(self, runner):

        print_log(f"epoch:{runner.epoch}, 剪枝前, 验证集精度", logger="current")
        runner.val_loop.run()

        if is_main_process():
            # 初始化tp
            self._init_tp(runner)

            if is_model_wrapper(runner.model):
                model = runner.model.module
            else:
                model = runner.model
            model.train()

            print_log(f"epoch:{runner.epoch}, 剪枝前, 网络结构", logger="current")
            print_log(model, logger="current")
            base_macs, base_nparams = tp.utils.count_ops_and_params(model, model.data_preprocessor(self.example_inputs, False))
            print_log(f"剪枝前, MACs({model.data_preprocessor(self.example_inputs, False)['inputs'].shape}): {base_macs/1e9} Billion, Params: {base_nparams/1e6} Million", logger="current")
            flops, _ = profile(model, inputs=(model.data_preprocessor(self.example_inputs, False)['inputs'],))
            print_log(f"剪枝前, FLOPs({model.data_preprocessor(self.example_inputs, False)['inputs'].shape}): {flops/1e9} Billion", logger="current")
            torch.onnx.export(model, model.data_preprocessor(self.example_inputs, False)['inputs'], osp.join(runner.work_dir, f"origin_model.onnx"), opset_version=11)

            print_log(f"开始剪枝......", logger="current")
            for g in self.pruner.step(interactive=True):
                print_log(g, logger="current")
                g.prune()

            print_log(f"epoch:{runner.epoch}, 剪枝后, 网络结构", logger="current")
            print_log(model, logger="current")
            base_macs_pruned, base_nparams_pruned = tp.utils.count_ops_and_params(model, model.data_preprocessor(self.example_inputs, False))
            print_log(f"剪枝后, MACs({model.data_preprocessor(self.example_inputs, False)['inputs'].shape}): {base_macs_pruned/1e9} Billion, Params: {base_nparams_pruned/1e6} Million", logger="current")
            flops, _ = profile(model, inputs=(model.data_preprocessor(self.example_inputs, False)['inputs'],))
            print_log(f"剪枝后, FLOPs({model.data_preprocessor(self.example_inputs, False)['inputs'].shape}): {flops/1e9} Billion", logger="current")

            print_log(f"保存剪枝后的模型", logger="current")
            torch.save(model, osp.join(runner.work_dir, f"pruned_model.pth"))
            torch.onnx.export(model, model.data_preprocessor(self.example_inputs, False)['inputs'], osp.join(runner.work_dir, f"pruned_model.onnx"), opset_version=11)

    def _foward_fn(self, model, data):
        data = model.data_preprocessor(data, True)
        return model._run_forward(data, mode='tensor')  # type: ignore
    
    def _init_tp(self, runner):
        print_log(f"初始化 Torch-Pruning, 使用 {self.pruning_algorithm} 算法", logger="current")
       
        # 准备数据
        print_log(f"从验证数据集中获取一个batch的数据, 用于剪枝trace", logger="current")
        for data_batch in runner.train_dataloader:
            self.example_inputs = data_batch
            break

        # 忽略层
        ignored_layers = []
        model = runner.model.module if is_distributed() else runner.model
        for name, module in model.named_modules():
            if self.ignore_layers_name is not None and any(i in name for i in self.ignore_layers_name):
                ignored_layers.append(module)
                print_log(f"忽略剪枝的层: {name}", logger="current")
                continue
            else:
                print_log(f"剪枝的层: {name}", logger="current")
        
        # 初始化剪枝器
        if self.pruning_algorithm == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        elif self.pruning_algorithm == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        elif self.pruning_algorithm == "l2":
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        elif self.pruning_algorithm == "fpgm":
            imp = tp.importance.FPGMImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        # elif self.pruning_algorithm == "obdc":
        #     imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=self.num_classes)
        #     pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        elif self.pruning_algorithm == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        # elif self.pruning_algorithm == "slim":
        #     self.sparsity_learning = True
        #     imp = tp.importance.BNScaleImportance()
        #     pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.reg, global_pruning=self.global_pruning)
        # elif self.pruning_algorithm == "group_slim":
        #     self.sparsity_learning = True
        #     imp = tp.importance.BNScaleImportance()
        #     pruner_entry = partial(tp.pruner.BNScalePruner, 
        #                            reg=self.reg, global_pruning=self.global_pruning, group_lasso=True)
        elif self.pruning_algorithm == "group_norm":
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=self.global_pruning)
        # elif self.pruning_algorithm == "group_sl":
        #     self.sparsity_learning = True
        #     imp = tp.importance.GroupNormImportance(p=2, normalizer='max')
        #     pruner_entry = partial(tp.pruner.GroupNormPruner, reg=self.reg, global_pruning=self.global_pruning)
        # elif self.pruning_algorithm == "growing_reg":
        #     self.sparsity_learning = True
        #     imp = tp.importance.GroupNormImportance(p=2)
        #     pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=self.reg, 
        #                            delta_reg=self.delta_reg, global_pruning=self.global_pruning)
        elif self.pruning_algorithm == "taylor":
            imp = tp.importance.TaylorImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.global_pruning)
        
        self.pruner = pruner_entry(
            model,
            self.example_inputs,
            importance=imp,
            iterative_steps=1,
            pruning_ratio=self.pruning_ratio,
            ignored_layers=ignored_layers,
            forward_fn=self._foward_fn,
            round_to=8,
            isomorphic=self.isomorphic,
        )
        print_log(f"初始化 Torch-Pruning 完成", logger="current")
