# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmyolo.models import BaseBackbone
from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP


@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """

    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)


@MODELS.register_module()
class MultiModalYOLOv8CSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv8.

    直接将原本的 YOLOv8 backbone 改造成multimodal backbone，便于用检测器的预训练权重
    原 YOLO-World 的 backbone 修改了结构，不便于直接使用检测器的预训练权重

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }

    def __init__(self,
                 text_model_config: ConfigType = None,
                 with_text_model: bool = True,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

        # 初始化 text encoder
        self.with_text_model = with_text_model
        if self.with_text_model:
            assert text_model_config is not None, "with_text_model is set to True, but model config is not provided!!!"
            self.text_model = MODELS.build(text_model_config)
        else:
            self.text_model = None

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

    def forward(self, image: Tensor, text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:

        image_outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            image = layer(image)
            if i in self.out_indices:
                image_outs.append(image)

        if self.with_text_model:
            text_outs = self.text_model(text)
            return tuple(image_outs), text_outs
        else:
            return tuple(image_outs), None

    def forward_image(self, image):

        image_outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            image = layer(image)
            if i in self.out_indices:
                image_outs.append(image)
        return tuple(image_outs)

    def forward_text(self, text):
        assert self.with_text_model, 'text model must be provided!!!'
        text_outs = self.text_model(text)
        return text_outs
