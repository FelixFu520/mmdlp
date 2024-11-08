# Copyright (c) FelixFu. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence, Union
import torch

import torch_pruning as tp
from mmengine.hooks import Hook
from mmdlp.registry import HOOKS
from mmengine.logging import print_log
from mmengine.device import get_device
from mmengine.model import is_model_wrapper
from mmengine.dist import master_only, is_main_process

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class MagnitudePruner(Hook):
    """
    使用剪枝算法对模型的通道进行剪枝, 以减少模型的计算量和参数量.
    参考: https://github.com/VainF/Torch-Pruning

    Args:
        自定义的参数
        by_epoch (bool): whether to prune by epoch or by iteration. Default: True.
        epoch_times (list): the epochs to prune. Default: None.
        iter_times (list): the iterations to prune. Default: None.
        pruning_rate (float): the ratio of channels to prune. Default: 0.5.
        pruning_algorithm (str): the pruning algorithm to use. Default: TaylorImportance.
        ignore_layers_name (list): the names of layers to ignore. Default: None.

        torch_pruning库中的参数, 下面的参数是MagnitudePruner的参数, 详细的参数说明可以参考torch_pruning的文档
        **kwargs: other keyword arguments for torch_pruning.pruner.MagnitudePruner
    """
    priority = 'VERY_LOW'


    def __init__(self, by_epoch: bool = True, epoch_times: list = None, iter_times: list = None, 
                 pruning_ratio: float = 0.5, ignore_layers_name: list = None,
                 pruning_lr_scale: float = 10, pruning_algorithm: str = "TaylorImportance", **kwargs):
        self.by_epoch = by_epoch
        self.epoch_times = epoch_times
        self.iter_times = iter_times
        if by_epoch:
            self.iterative_steps = len(epoch_times)
        else:
            self.iterative_steps = len(iter_times)
        if by_epoch:
            assert epoch_times is not None, "epoch_times should be provided if by_epoch is True"
        else:
            assert iter_times is not None, "iter_times should be provided if by_epoch is False"

        self.pruning_ratio = pruning_ratio
        assert self.pruning_ratio > 0 and self.pruning_ratio < 1, "pruning_ratio should be in (0, 1)"

        self.pruning_lr_scale = pruning_lr_scale
        
        self.pruning_algorithm = pruning_algorithm
        if pruning_algorithm == "TaylorImportance":
            self.importance = tp.importance.TaylorImportance()
        elif pruning_algorithm == "MagnitudeImportance":
            self.importance = tp.importance.MagnitudeImportance()
        elif pruning_algorithm == "HessianImportance":
            self.importance = tp.importance.HessianImportance()
        else:
            raise ValueError(f"Unsupported pruning algorithm: {pruning_algorithm}")
        self.ignore_layers_name = ignore_layers_name
        self.kwargs = kwargs
        self.is_inited = False
        
    def before_train_epoch(self, runner):
        if self.by_epoch and runner.epoch in self.epoch_times:
            # 初始化tp
            if not self.is_inited:
                self._init_tp(runner)
                self.is_inited = True

            if is_model_wrapper(runner.model):
                model = runner.model.module
            else:
                model = runner.model
        
            print_log(f"epoch:{runner.epoch}, 剪枝前, 验证集精度", logger="current")
            runner.val_loop.run()

            print_log(f"epoch:{runner.epoch}, 剪枝前, 网络结构", logger="current")
            print_log(model, logger="current")

            base_macs, base_nparams = tp.utils.count_ops_and_params(model, model.data_preprocessor(self.example_inputs, False))
            print_log(f"剪枝前, MACs: {base_macs/1e6} M, Params: {base_nparams/1e6} M", logger="current")

            print_log(f"开始剪枝......", logger="current")
            loss = model(model.data_preprocessor(self.example_inputs, False)['inputs']).sum()
            loss.backward()
            self.pruner.step()

            print_log(f"epoch:{runner.epoch}, 剪枝后, 网络结构", logger="current")
            print_log(model, logger="current")

            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, model.data_preprocessor(self.example_inputs, False))
            print_log(f"剪枝后, MACs: {pruned_macs/1e6} M, Params: {pruned_nparams/1e6} M", logger="current")

    def after_train(self, runner):
        if is_main_process():
            print_log(f"保存剪枝后的模型", logger="current")
            if is_model_wrapper(runner.model):
                model = runner.model.module
            else:
                model = runner.model
            torch.save(model, osp.join(runner.work_dir, f"pruned_model.pth"))
            torch.onnx.export(model, model.data_preprocessor(self.example_inputs, False)['inputs'], osp.join(runner.work_dir, f"pruned_model.onnx"))
    
    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        if not self.by_epoch and (runner.iter + 1) in self.iter_times:
            if not self.is_inited:
                self._init_tp(runner)
                self.is_inited = True
            raise NotImplementedError("Pruning by iteration is not implemented yet")
    
    def _foward_fn(self, model, data):
        data = model.data_preprocessor(data, False)
        return model._run_forward(data, mode='tensor')  # type: ignore
    
    def _init_tp(self, runner):
        if self.is_inited:
            return
        
        # 获得model
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        
        print_log(f"初始化 Torch-Pruning, 使用 {self.pruning_algorithm} 算法", logger="current")
       
        # 准备数据
        print_log(f"从验证数据集中获取一个batch的数据, 用于剪枝trace", logger="current")
        for data_batch in runner.val_dataloader:
            self.example_inputs = data_batch
            self.example_inputs['inputs'] = self.example_inputs['inputs'].to(get_device())
            break

        # 忽略层
        ignored_layers = []
        for name, module in model.named_modules():
            if self.ignore_layers_name is not None and name in self.ignore_layers_name:
                ignored_layers.append(module)
                print_log(f"忽略剪枝的层: {name}", logger="current")
                continue
            else:
                print_log(f"剪枝的层: {name}", logger="current")
        
        # 初始化剪枝器
        self.pruner = tp.pruner.MagnitudePruner(
            model,
            self.example_inputs,
            importance=self.importance,
            iterative_steps=self.iterative_steps,
            ch_sparsity=self.pruning_ratio,
            ignored_layers=ignored_layers,
            forward_fn=self._foward_fn,
        )

        # 修改optime的学习率
        if self.pruning_lr_scale != 1:
            for param_group in runner.optim_wrapper.param_groups:
                param_group['lr'] *= self.pruning_lr_scale
                print_log(f"剪枝时, 学习率乘以 {self.pruning_lr_scale}, 新的学习率: {param_group['lr']}", logger="current")