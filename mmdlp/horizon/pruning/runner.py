# Copyright (c) FelixFu. All rights reserved.
import logging
from typing import Callable, Dict, List, Optional, Sequence, Union
import warnings
import time

import torch
import torch.nn as nn

import mmengine
from mmengine import print_log
from mmengine.dist import master_only
from mmengine.fileio import FileClient, join_path
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.optim import OptimWrapper
from mmengine.utils import apply_to, get_git_hash
from mmengine.device import get_device
from mmengine.dist import cast_data_device
from mmengine.runner import weights_to_cpu, save_checkpoint, Runner
from mmdlp.registry import RUNNERS


def _fix_special(model):
    if get_device() == 'cuda':
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        cast_model = cast_data_device(model.bbox_head.mlvl_priors, device)
        model.bbox_head.mlvl_priors = cast_model
    

@RUNNERS.register_module()
class PruningRunner(Runner):
    """继承自mmengine.runner.Runner, 主要是为了配合剪枝使用, 修改以下函数
    - build_model
    - save_checkpoint
    主要是因为, 原Runner中只保存权重, 不保存网络结构, 但是剪枝需要保存网络结构.
    """
   
    def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
        """直接从模型文件中加载模型, 并返回模型
        Args:
            model (Union[nn.Module, Dict]): 模型或者模型的配置文件
        """
        assert not self._resume, "禁止使用resume"
        assert not self._load_from, "禁止使用load_from"

        if isinstance(model, nn.Module):
            print_log(model, logger="current")
            return model
        elif isinstance(model, dict):
            model_path = model.pop('path')
            model = torch.load(model_path, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))
            print_log(model, logger="current")
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')
        
    @master_only
    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: Optional[dict] = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Decide the number of epoch or iteration saved in
                checkpoint. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.setdefault('epoch', self.epoch + 1)
            meta.setdefault('iter', self.iter)
        else:
            meta.setdefault('epoch', self.epoch)
            meta.setdefault('iter', self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta':
            meta,
            'model':
            model,
            'state_dict':
            weights_to_cpu(model.state_dict()),
            'message_hub':
            apply_to(self.message_hub.state_dict(),
                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = apply_to(
                    self.optim_wrapper.state_dict(),
                    lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            self.logger.warning(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers')
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(
            checkpoint,
            filepath,
            file_client_args=file_client_args,
            backend_args=backend_args)
