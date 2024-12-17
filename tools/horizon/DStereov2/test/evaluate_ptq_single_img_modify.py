
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
from hmct.ir import load_model, save_model
# from hat.data.datasets.multi_disp_dataset.list_dataset import Instereo2KDataset
# from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset
from horizon_tc_ui import HB_ONNXRuntime
# from horizon_tc_ui.data.transformer import *
# from horizon_nn.common import modify_model_by_cpp_func
# from horizon_nn.ir import load_model, save_model
# from horizon_nn.ir.horizon_onnx import global_attributes, quant_attributes, quantizer
# from horizon_nn.tools.compare_calibrated_and_quantized_model import ConsistencyChecker

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
    assert key in ["original_float_model", "optimized_float_model", "calibrated_model_test", "quantized_model"], key
    if key == "quantized_model":
        left -= 128
        infra1 = left.astype(np.int8)
        right -= 128
        infra2 = right.astype(np.int8)
    elif key in ["original_float_model", "optimized_float_model", "calibrated_model_test"]:
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
        os.path.join(exp_root, "%s_calibrated_model_test.onnx" % onnx_prefix),
        os.path.join(exp_root, "%s_quantized_model.onnx" % onnx_prefix),
    ]
    calibrated_model = load_model(os.path.join(exp_root, "%s_calibrated_model.onnx" % onnx_prefix))
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        # if node.tensor_type == "weight":
        #     node.qtype = "int16"
        # if node.tensor_type == "featuremap":
        #     node.qtype = "int16"
        if node.name in [
            "onnx::Conv_2747_HzCalibration",
            "onnx::Conv_2864_HzCalibration",
            "onnx::Conv_2750_HzCalibration",
            "/get_initdisp/GEMMvariable_2921_conv_weight_HzCalibration",
            "onnx::Conv_2867_HzCalibration",
            "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1/Conv_HzCalibration",
            "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1_1/Conv_HzCalibration",
            # "/get_initdisp/GEMMvariable_2921_conv_weight_HzCalibration",
            # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1_1/Conv_HzCalibration",
            # "refinement.update_block.gru.convq.weight_/refinement/update_block/gru/convq_1/Conv_HzCalibration",
            # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1/Conv_HzCalibration",
            # "onnx::Conv_3071_HzCalibration",
            # "onnx::Conv_2747_HzCalibration",
            # "refinement.update_block.encoder.convd2.weight_/refinement/update_block/encoder/convd2_1/Conv_HzCalibration",
            # "onnx::Conv_2750_HzCalibration",
            # "refinement.spx_gru.0.weight_HzCalibration",
            # "refinement.update_block.mask_feat_4.0.weight_HzCalibration",
            # "refinement.update_block.gru.convq.weight_/refinement/update_block/gru/convq/Conv_HzCalibration",
        ]:
            node.qtype = "int16"
    save_model(calibrated_model, os.path.join(exp_root, "%s_calibrated_model_test.onnx" % onnx_prefix))

    validation_dict = {}
    onnx_infer_dict = {}
    for onnx_file in target_onnx_list:
        key = onnx_file.replace(os.path.join(exp_root, onnx_prefix + "_"), "").replace(".onnx", "")
        if key not in validation_dict.keys():
            validation_dict[key] = {"out_list": [], "epe_list": []}
        sess = HB_ONNXRuntime(model_file=onnx_file)
        input_names = [input.name for input in sess.get_inputs()]
        output_names = [output.name for output in sess.get_outputs()]
        if key not in onnx_infer_dict.keys():
            onnx_infer_dict[key] = {"sess": sess, "input_names": input_names, "output_names": output_names}

    val = (disp_gt.flatten() >= 0.01) & (disp_gt.abs().flatten() < 192)

    for onnx_file in target_onnx_list:
        key = onnx_file.replace(os.path.join(exp_root, onnx_prefix + "_"), "").replace(".onnx", "")
        disp_pr = get_onnx_infer_result(key, onnx_infer_dict, copy.deepcopy(left), copy.deepcopy(right))
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
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


if __name__ == '__main__':
    left = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_left.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
    right = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_right.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
    disp_gt = torch.from_numpy(np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_disp_gt.npy", dtype=np.float32).reshape(352, 640))
    exp_root = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v1/"
    onnx_prefix = 'PTQ_check_yuv444'
    validate_instereo2k(exp_root, onnx_prefix, left, right, disp_gt)
