
import sys
import os
import argparse
import time
import logging
import os.path as osp
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt
import copy
# from hat.data.datasets.multi_disp_dataset.list_dataset import Instereo2KDataset
# from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset
from horizon_tc_ui import HB_ONNXRuntime
# from horizon_tc_ui.data.transformer import *
# from horizon_nn.common import modify_model_by_cpp_func
# from horizon_nn.ir import load_model, save_model
# from horizon_nn.ir.horizon_onnx import global_attributes, quant_attributes, quantizer
# from horizon_nn.tools.compare_calibrated_and_quantized_model import ConsistencyChecker

def disp2rgb(disp, disp_max, disp_min):
    mask = np.logical_or(disp > disp_max, disp < disp_min)
    disp = np.clip(disp, disp_min, disp_max)  # 近处是0， 远处320
    mat_min = disp.min()
    mat_max = disp.max()
    norm_matrix = (disp - mat_min) / (mat_max - mat_min)
    disp = 0.1 + norm_matrix * (0.9 - 0.1)
    # disp = disp / 355
    # disp = disp / disp.max()
    # disp = abs(disp - 1)
    disp *= 256
    disp = np.round(disp).astype(np.uint8)[..., None]
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)#[:,:,::-1]
    disp[mask] = (0, 0, 0)

    return disp

def validate_result(disp_pr, disp_gt):
    assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
    epe = torch.abs(disp_pr - disp_gt)

    epe = epe.flatten()
    out = (epe > 3.0)

    return epe, out

def result2disp(disp_unfold, spx):
    disp = torch.sum(disp_unfold*spx, 1, keepdim=False)[0]
    return disp

def get_onnx_infer_result(key, onnx_infer_dict, left, right, lossy=False):
    assert key in ["original_float_model", "optimized_float_model", "calibrated_model", "quantized_model"], key
    if key == "quantized_model":
        left -= 128
        infra1 = left.astype(np.int8)
        right -= 128
        infra2 = right.astype(np.int8)
    elif key in ["original_float_model", "optimized_float_model", "calibrated_model"]:
        infra1 = left.transpose(0, 3, 1, 2).astype(np.float32)
        infra2 = right.transpose(0, 3, 1, 2).astype(np.float32)
    
    feed_dict = {
        onnx_infer_dict[key]['input_names'][0]: np.ascontiguousarray(infra1),
        onnx_infer_dict[key]['input_names'][1]: np.ascontiguousarray(infra2),
    }

    disp_unfold, spx = onnx_infer_dict[key]['sess'].run(onnx_infer_dict[key]['output_names'][:2], feed_dict)
    disp_unfold = torch.from_numpy(disp_unfold)
    spx = torch.from_numpy(spx)
    outputs = result2disp(disp_unfold, spx)
    return outputs.contiguous()

def validate_instereo2k(exp_root, onnx_prefix, left, right, disp_gt):
    result_save_root = os.path.join(exp_root, "infer_result")
    os.makedirs(result_save_root, exist_ok=True)
    target_onnx_list = [
        os.path.join(exp_root, "%s_original_float_model.onnx" % onnx_prefix),
        os.path.join(exp_root, "%s_optimized_float_model.onnx" % onnx_prefix),
        os.path.join(exp_root, "%s_calibrated_model.onnx" % onnx_prefix),
        os.path.join(exp_root, "%s_quantized_model.onnx" % onnx_prefix),
    ]

    validation_dict = {}
    onnx_infer_dict = {}
    for onnx_file in target_onnx_list:
        key = onnx_file.replace(os.path.join(exp_root, onnx_prefix + "_"), "").replace(".onnx", "")
        if key not in validation_dict.keys():
            validation_dict[key] = {"out_list": [], "epe_list": [], "disp": None}
        sess = HB_ONNXRuntime(model_file=onnx_file)
        input_names = [input.name for input in sess.get_inputs()]
        output_names = [output.name for output in sess.get_outputs()]
        if key not in onnx_infer_dict.keys():
            onnx_infer_dict[key] = {"sess": sess, "input_names": input_names, "output_names": output_names}

    val = (disp_gt.flatten() >= 0.01) & (disp_gt.abs().flatten() < 192)

    for onnx_file in target_onnx_list:
        key = onnx_file.replace(os.path.join(exp_root, onnx_prefix + "_"), "").replace(".onnx", "")
        disp_pr = get_onnx_infer_result(key, onnx_infer_dict, copy.deepcopy(left), copy.deepcopy(right))
        validation_dict[key]['disp'] = copy.deepcopy(disp_pr.cpu().numpy())
        epe, out = validate_result(disp_pr, disp_gt)
        if(np.isnan(epe[val].mean().item())):
            raise NotImplementedError

        validation_dict[key]['epe_list'].append(epe[val].mean().item())
        validation_dict[key]['out_list'].append(out[val].cpu().numpy())
        
    for onnx_file in target_onnx_list:
        key = onnx_file.replace(os.path.join(exp_root, onnx_prefix + "_"), "").replace(".onnx", "")

        epe_list = np.array(validation_dict[key]['epe_list'])
        out_list = np.concatenate(validation_dict[key]['out_list'])

        epe = np.mean(epe_list)
        d1 = 100 * np.mean(out_list)

        print("%s Validation Instereo2K: %f, %f" % (key, epe, d1))
    
    for key in ["calibrated_model", "quantized_model"]:
        disp_pr_diff = np.abs(validation_dict[key]['disp'] - validation_dict["original_float_model"]['disp'])
        disp_pr_diff_rgb = disp2rgb(disp_pr_diff, disp_pr_diff.max().item(), 0)
        cv2.imwrite("%s_disp_diff.png" % key, disp_pr_diff_rgb)
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}



if __name__ == '__main__':

    left = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_left.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
    right = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_right.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
    disp_gt = torch.from_numpy(np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_disp_gt.npy", dtype=np.float32).reshape(352, 640))
    exp_root = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v12/"
    onnx_prefix = 'PTQ_check_yuv444'
    validate_instereo2k(exp_root, onnx_prefix, left, right, disp_gt)