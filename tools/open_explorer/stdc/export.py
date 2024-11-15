import os
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot

DEVICE = "cuda:4"
IS_WRAPPER = False
HEIGHT=480
WIDTH=640
fake_input = torch.randn(1, 3, HEIGHT, WIDTH).to(DEVICE)

if IS_WRAPPER:
    onnx_paths = [
        "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx/STDC1_nopre_wrapper.onnx",
        "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx/STDC1_pre_wrapper.onnx",
        "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx/STDC2_nopre_wrapper.onnx",
        "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx/STDC2_pre_wrapper.onnx",
    ]
else:
    onnx_paths = [
        f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_{HEIGHT}×{WIDTH}/STDC1_nopre.onnx",
        f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_{HEIGHT}×{WIDTH}/STDC1_pre.onnx",
        f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_{HEIGHT}×{WIDTH}/STDC2_nopre.onnx",
        f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_{HEIGHT}×{WIDTH}/STDC2_pre.onnx",
    ]
config_files = [
    '/home/users/fa.fu/work/mmdlp/configs/stdc_horizon/config_from_mmseg/stdc/stdc1_4xb12-80k_cityscapes-512x1024.py',
    '/home/users/fa.fu/work/mmdlp/configs/stdc_horizon/config_from_mmseg/stdc/stdc1_in1k-pre_4xb12-80k_cityscapes-512x1024.py',
    '/home/users/fa.fu/work/mmdlp/configs/stdc_horizon/config_from_mmseg/stdc/stdc2_4xb12-80k_cityscapes-512x1024.py',
    '/home/users/fa.fu/work/mmdlp/configs/stdc_horizon/config_from_mmseg/stdc/stdc2_in1k-pre_4xb12-80k_cityscapes-512x1024.py',
]
checkpoint_files = [
    '/home/users/fa.fu/work/work_dirs/stdc_horizon/stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth',
    '/home/users/fa.fu/work/work_dirs/stdc_horizon/stdc1_in1k-pre_512x1024_80k_cityscapes_20220224_141648-3d4c2981.pth',
    '/home/users/fa.fu/work/work_dirs/stdc_horizon/stdc2_512x1024_80k_cityscapes_20220222_132015-fb1e3a1a.pth',
    '/home/users/fa.fu/work/work_dirs/stdc_horizon/stdc2_in1k-pre_512x1024_80k_cityscapes_20220224_073048-1f8f0f6c.pth'
]
img = '/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/demo_images/krefeld_000000_012353_leftImg8bit.png'

class WraperStdcUpsample(nn.Module):
    def __init__(self, stdc: nn.Module):
        super(WraperStdcUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
        self.stdc = stdc

    def forward(self, x):
        x = self.stdc(x)
        x = self.upsample(x)
        x = F.softmax(x, dim=1)
        score, label = torch.max(x, dim=1)
        return score, label
    

if __name__ == "__main__":
    for i, (onnx_path, config_file, checkpoint_file) in enumerate(zip(onnx_paths, config_files, checkpoint_files)):

        # build the model from a config file and a checkpoint file
        model = init_model(config_file, checkpoint_file, device=DEVICE)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        # export the model to onnx
        if IS_WRAPPER:
            wrapper_model = WraperStdcUpsample(model).eval()
            torch.onnx.export(
                wrapper_model,
                fake_input,
                onnx_path,
                opset_version=11,
                input_names=["input"],
                output_names=["score", "label"],
                verbose=True,
                do_constant_folding=True, 
                export_params=True, 
            )
        else:
            wrapper_model = model.eval()
            torch.onnx.export(
                wrapper_model,
                fake_input,
                onnx_path,
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
                verbose=True, 
                do_constant_folding=True, 
                export_params=True, 
            )
        model_opt, check_ok =  onnxsim.simplify(onnx_path)
        onnx.save(model_opt, onnx_path[:-5] + "_opt.onnx")

        # test a single image
        if not torch.cuda.is_available():
            model = revert_sync_batchnorm(model)
        result = inference_model(model, img)

        # show the results
        vis_result = show_result_pyplot(model, img, result, show=False, save_dir=f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx/{i}")
        plt.imshow(vis_result)