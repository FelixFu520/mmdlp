# stdc模型输入为1024*2048时, 地瓜的x5芯片不可以推理, 因为超过了芯片的限制.
# 由于在 1024x2048 分辨率下存一个 F.interpolate (ONNX Resize) 对 Tensor 进行上采样,
# 由 (1, 128, 1, 1)变为(1，128，32，64),
# 其中 output_shape[-1] / input_shape[-1]  factor= 64/1 超出编译器算子约束限制, 会将该算子回退 CPU 从而影响性能.
# factor_H CHECK failed: 64 not in the range of { 1   2   4   8   16   32   } 

# 定位方法: 可以通过OE工具链的hdk编译器的编译日志查看, 找到有问题的onnx算子, 然后在onnx中找到算子的代码位置,
# 然后再对源码进行修改, 下面就是修改过程
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList
from mmseg.models.utils import resize
from mmcv.cnn import ConvModule
from mmseg.models.backbones.bisenetv1 import AttentionRefinementModule
from mmseg.models.backbones.stdc import FeatureFusionModule
from mmdlp.registry import MODELS


@MODELS.register_module()
class STDCContextPathNet(BaseModule):
    """STDCNet with Context Path. The `outs` below is a list of three feature
    maps from deep to shallow, whose height and width is from small to big,
    respectively. The biggest feature map of `outs` is outputted for
    `STDCHead`, where Detail Loss would be calculated by Detail Ground-truth.
    The other two feature maps are used for Attention Refinement Module,
    respectively. Besides, the biggest feature map of `outs` and the last
    output of Attention Refinement Module are concatenated for Feature Fusion
    Module. Then, this fusion feature map `feat_fuse` would be outputted for
    `decode_head`. More details please refer to Figure 4 of original paper.

    Args:
        backbone_cfg (dict): Config dict for stdc backbone.
        last_in_channels (tuple(int)), The number of channels of last
            two feature maps from stdc backbone. Default: (1024, 512).
        out_channels (int): The channels of output feature maps.
            Default: 128.
        ffm_cfg (dict): Config dict for Feature Fusion Module. Default:
            `dict(in_channels=512, out_channels=256, scale_factor=4)`.
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``.
        align_corners (str): align_corners argument of F.interpolate. It
            must be `None` if upsample_mode is ``'nearest'``. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Return:
        outputs (tuple): The tuple of list of output feature map for
            auxiliary heads and decoder head.
    """

    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in last_in_channels:
            self.arms.append(AttentionRefinementModule(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))
        self.conv_avg = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)

        self.ffm = FeatureFusionModule(**ffm_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = list(self.backbone(x))
        avg = F.adaptive_avg_pool2d(outs[-1], 1)
        avg_feat = self.conv_avg(avg)

        outs_shape = outs[-1].shape[2:]
        if outs[-1].shape[-1] / avg_feat.shape[-1] > 32:
                # 适配 X5 
                feature_up0 = resize(
                        avg_feat,
                        size=[outs_shape[0]//2, outs_shape[1]//2],
                        mode=self.upsample_mode,
                        align_corners=self.align_corners)
                feature_up = resize(
                        feature_up0,
                        size=[outs_shape[0], outs_shape[1]],
                        mode=self.upsample_mode,
                        align_corners=self.align_corners)
        else:   
            feature_up = resize(
                avg_feat,
                size=outs[-1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
        
        arms_out = []
        for i in range(len(self.arms)):
            x_arm = self.arms[i](outs[len(outs) - 1 - i]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[len(outs) - 1 - i - 1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        feat_fuse = self.ffm(outs[0], arms_out[1])

        # The `outputs` has four feature maps.
        # `outs[0]` is outputted for `STDCHead` auxiliary head.
        # Two feature maps of `arms_out` are outputted for auxiliary head.
        # `feat_fuse` is outputted for decoder head.
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)
