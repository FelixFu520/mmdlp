
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
from hmct.common import find_input_calibration, find_output_calibration
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
    # calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    # for node in calibration_nodes:
    #     if node.tensor_type == "weight":
    #         node.qtype = "int16"
    #     if node.tensor_type == "feature":
    #         node.qtype = "int16"
    #     if node.name in [
    #         # "/get_initdisp/GEMM_split0",
    #         # "/get_initdisp/GEMM_split1",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.1/conv.1.0/Conv",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2/Relu",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv_split_add0",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.1/conv.1.0_1/Conv",
    #         # "/refinement/update_block/encoder/convd1_1/Conv_split0",
    #         # "/refinement/update_block/encoder/convd1_1/Conv_split1",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2_1/Relu",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0_1/Conv_split_add0",
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.0/LeakyRelu",
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.1/Conv",
    #         # "/backbone/mod6/mod6.0/head_layer/Add",
    #         # "/backbone/mod6/mod6.0/stack_layers/stack_layers.0/Add",
    #         # "/backbone/mod6/mod6.0/head_layer/relu/Relu",
    #         # "/refinement/update_block/encoder/convd1/Conv_split0",
    #         # "/refinement/update_block/encoder/convd1/Conv_split1",
    #         # "/get_initdisp/classifier/LeakyRelu",
    #         # "/cost_agg/conv1/conv1.0/LeakyRelu",
    #         # "/cost_agg/conv1/conv1.1/conv/Conv",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv_split0",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv_split1",
    #         # "/get_initdisp/Softmax_reducemax_FROM_QUANTIZED_SOFTMAX",
    #         # "/get_initdisp/Softmax_sub_FROM_QUANTIZED_SOFTMAX",
    #         # "/cost_agg/conv2/conv2.1/LeakyRelu",
    #         # "/cost_agg/feature_att_16/Mul",
    #         # "/backbone/mod3/mod3.0/head_layer/downsample/downsample.0/Conv",
    #         # "/refinement/update_block/gru/convq/Conv",
    #         # "/get_initdisp/Softmax",
    #         # "/refinement/Softmax",
    #         # "/refinement/interp_conv/depth2space_1/DepthToSpace",

    #         # "/get_initdisp/GEMM_/get_initdisp/GEMM_pre_reshape_output_transpose_in_calibrated_HzCalibration",
    #         # "variable_1386_HzCalibration",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2/Relu_output_0_calibrated_HzCalibration",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2/Relu_output_0_HzCalibration",
    #         # "variable_1384_HzCalibration",
    #         # "variable_1380_HzCalibration",
    #         # "variable_1380_calibrated_HzCalibration",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2_1/Relu_output_0_calibrated_HzCalibration",
    #         # "/refinement/Add_output_0_calibrated_HzCalibration",
    #         # "variable_1396_HzCalibration",
    #         # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.2_1/Relu_output_0_HzCalibration",
    #         # "variable_1392_HzCalibration",
    #         # "variable_1392_calibrated_HzCalibration"
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.0/conv/Conv_output_0_HzCalibration",
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.0/LeakyRelu_output_0_HzCalibration",
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.0/conv/Conv_output_0_calibrated_HzCalibration",
    #         # "/cost_agg/feature_att_8/feat_att/feat_att.0/conv/conv.1/conv.1.0/Conv_output_0_HzCalibration",
    #         # "/backbone/mod6/mod6.0/head_layer/conv/conv.1/conv.1.0/Conv_output_0_HzCalibration",
    #         # "/backbone/mod6/mod6.0/head_layer/relu/Relu_output_0_HzCalibration",
    #         # "/backbone/mod6/mod6.0/head_layer/Add_output_0_HzCalibration",
    #         # "/get_initdisp/classifier/conv/Conv_output_0_HzCalibration",
    #         # "variable_2926_calibrated_HzCalibration",
    #         # "/get_initdisp/classifier/LeakyRelu_output_0_HzCalibration",
    #         # "/get_initdisp/classifier/conv/Conv_output_0_calibrated_HzCalibration",
    #         # "/backbone/mod6/mod6.0/head_layer/conv/conv.1/conv.1.0/Conv_output_0_calibrated_HzCalibration",
    #         # "/cost_agg/conv1/conv1.0/conv/Conv_output_0_HzCalibration",
    #         # "/cost_agg/conv1/conv1.0/LeakyRelu_output_0_HzCalibration",
    #         # "/cost_agg/agg_1/agg_1.0/conv/Conv_output_0_calibrated_HzCalibration",
    #         # "/cost_agg/conv2/conv2.1/conv/Conv_output_0_HzCalibration",
    #         # "/cost_agg/conv2/conv2.0/conv/Conv_output_0_HzCalibration",
    #         # "variable_1372_calibrated_HzCalibration",
    #         # "/cost_agg/conv2/conv2.1/conv/Conv_output_0_calibrated_HzCalibration",
    #         # "/cost_agg/conv2/conv2.1/LeakyRelu_output_0_HzCalibration",

    #         # "onnx::Conv_2747_HzCalibration",
    #         # "onnx::Conv_2864_HzCalibration",
    #         # "onnx::Conv_2750_HzCalibration",
    #         # "/get_initdisp/GEMMvariable_2921_conv_weight_HzCalibration",
    #         # "onnx::Conv_2867_HzCalibration",
    #         # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1/Conv_HzCalibration",
    #         # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1_1/Conv_HzCalibration",
    #         # "/get_initdisp/GEMMvariable_2921_conv_weight_HzCalibration",
    #         # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1_1/Conv_HzCalibration",
    #         # "refinement.update_block.gru.convq.weight_/refinement/update_block/gru/convq_1/Conv_HzCalibration",
    #         # "refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1/Conv_HzCalibration",
    #         # "onnx::Conv_3071_HzCalibration",
    #         # "onnx::Conv_2747_HzCalibration",
    #         # "refinement.update_block.encoder.convd2.weight_/refinement/update_block/encoder/convd2_1/Conv_HzCalibration",
    #         # "onnx::Conv_2750_HzCalibration",
    #         # "refinement.spx_gru.0.weight_HzCalibration",
    #         # "refinement.update_block.mask_feat_4.0.weight_HzCalibration",
    #         # "refinement.update_block.gru.convq.weight_/refinement/update_block/gru/convq/Conv_HzCalibration",
    #     ]:
    #         node.qtype = "int16"
           
    for node in calibrated_model.graph.nodes:
        if node.name in [
            "/backbone/mod6/mod6.0/head_layer/downsample/downsample.0_1/Conv",
            "/backbone/mod6/mod6.0/head_layer/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod6/mod6.0/head_layer/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod6/mod6.0/head_layer/downsample/downsample.0/Conv",
            "/backbone/mod6/mod6.0/head_layer/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod6/mod6.0/head_layer/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod6/mod6.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0/Conv",


            "/backbone/mod5/mod5.0/head_layer/downsample/downsample.0_1/Conv",
            "/backbone/mod5/mod5.0/head_layer/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/head_layer/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/head_layer/downsample/downsample.0/Conv",
            "/backbone/mod5/mod5.0/head_layer/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/head_layer/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.2/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.2/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.3/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.3/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.4/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.4/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.5/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.5/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.2/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.2/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.3/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.3/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.4/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.4/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.5/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod5/mod5.0/stack_layers/stack_layers.5/conv/conv.1/conv.1.0/Conv",

            "/backbone/mod4/mod4.0/head_layer/downsample/downsample.0_1/Conv",
            "/backbone/mod4/mod4.0/head_layer/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod4/mod4.0/head_layer/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod4/mod4.0/head_layer/downsample/downsample.0/Conv",
            "/backbone/mod4/mod4.0/head_layer/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod4/mod4.0/head_layer/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod4/mod4.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0/Conv",

            "/backbone/mod3/mod3.0/head_layer/downsample/downsample.0_1/Conv",
            "/backbone/mod3/mod3.0/head_layer/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod3/mod3.0/head_layer/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0_1/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0_1/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0_1/Conv",

            "/backbone/mod2/mod2.0/head_layer/conv/conv.1/conv.1.0_1/Conv",

            "/feature/deconv32_16/conv2/conv/Conv",
            "/feature/deconv16_8/conv2/conv/Conv",
            "/feature/deconv8_4/conv2/conv/Conv",
            "/feature/conv4/conv/Conv",
            "/feature/deconv32_16/conv2/conv_1/Conv",
            "/feature/deconv16_8/conv2/conv_1/Conv",
            "/feature/deconv8_4/conv2/conv_1/Conv",
            "/feature/conv4/conv_1/Conv",

            "/before_costvolum/conv/conv/Conv",
            "/before_costvolum/desc/Conv",
            "/before_costvolum/conv/conv_1/Conv",
            "/before_costvolum/desc_1/Conv",

            "/cost_agg/conv1/conv1.0/conv/Conv",
            "/cost_agg/conv1/conv1.1/conv/Conv",
            "/cost_agg/feature_att_8/feat_att/feat_att.0/conv/Conv",
            "/cost_agg/feature_att_8/feat_att/feat_att.1/Conv",
            "/cost_agg/conv2/conv2.0/conv/Conv",
            "/cost_agg/conv2/conv2.1/conv/Conv",
            "/cost_agg/feature_att_16/feat_att/feat_att.0/conv/Conv",
            "/cost_agg/feature_att_16/feat_att/feat_att.1/Conv",
            "/cost_agg/conv3/conv3.0/conv/Conv",
            "/cost_agg/conv3/conv3.1/conv/Conv",
            "/cost_agg/feature_att_32/feat_att/feat_att.0/conv/Conv",
            "/cost_agg/feature_att_32/feat_att/feat_att.1/Conv",
            "/cost_agg/agg_0/agg_0.0/conv/Conv",
            "/cost_agg/agg_0/agg_0.1/conv/Conv",
            "/cost_agg/agg_0/agg_0.2/conv/Conv",
            "/cost_agg/feature_att_up_16/feat_att/feat_att.0/conv/Conv",
            "/cost_agg/feature_att_up_16/feat_att/feat_att.1/Conv",
            "/cost_agg/agg_1/agg_1.0/conv/Conv",
            "/cost_agg/agg_1/agg_1.1/conv/Conv",
            "/cost_agg/agg_1/agg_1.2/conv/Conv",
            "/cost_agg/feature_att_up_8/feat_att/feat_att.0/conv/Conv",
            "/cost_agg/feature_att_up_8/feat_att/feat_att.1/Conv",

            "/prepare_forrefinement/hnet/hnet.0/conv/Conv",
            "/prepare_forrefinement/hnet/hnet.1/Conv",
            "/prepare_forrefinement/cnet/conv/Conv",
            "/prepare_forrefinement/context_zqr_conv/Conv",

            "/refinement/update_block/encoder/convc1/Conv",
            "/refinement/update_block/encoder/convc2/Conv",
            "/refinement/update_block/encoder/convd2/Conv",
            "/refinement/update_block/encoder/conv/Conv",
            "/refinement/update_block/gru/convz/Conv",
            "/refinement/update_block/gru/convr/Conv",
            "/refinement/update_block/gru/convq/Conv",
            "/refinement/update_block/disp_head/conv1/Conv",
            "/refinement/update_block/disp_head/conv2/Conv",


            # 第二次添加
            "/get_initdisp/classifier/conv/Conv",
            "/refinement/update_block/encoder/convd2_1/Conv",
            "/refinement/update_block/gru/convz_1/Conv",
            "/refinement/update_block/encoder/conv_1/Conv",
            "/refinement/update_block/gru/convr_1/Conv",
            "/refinement/update_block/disp_head/conv1_1/Conv",
            "/refinement/update_block/disp_head/conv2_1/Conv",
            "/refinement/update_block/mask_feat_4/mask_feat_4.0/Conv",
            "/refinement/spx_2_gru/conv2/conv/Conv",
            "/refinement/update_block/encoder/convd1/Conv_split1",
            "/refinement/update_block/encoder/convd1/Conv_split0",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.0/conv/conv.0/conv.0.0/Conv",


            # 容忍误差较小 0.91
            "/refinement/update_block/encoder/convd1_1/Conv_split1",
            "/refinement/update_block/encoder/convd1_1/Conv_split0",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.1/conv/conv.0/conv.0.0/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.1/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod3/mod3.0/stack_layers/stack_layers.0/conv/conv.1/conv.1.0/Conv",
            "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv_split1",
            "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv_split0",
            "/backbone/mod3/mod3.0/head_layer/conv/conv.1/conv.1.0/Conv",

            # 容忍误差较大 0.92
            "/backbone/mod1/mod1.0_1/Conv_split1",
            "/backbone/mod1/mod1.0_1/Conv_split0",
            "/backbone/mod1/mod1.0/Conv_split1",
            "/backbone/mod1/mod1.0/Conv_split0",

            # Bad 0.93 +
            # "/refinement/Mul",
            # "/refinement/update_block/gru/convq_1/Conv",
            # "/refinement/unfold_conv/unflod_conv/Conv",
            # "/refinement/interp_conv/conv/Conv",
            # "/refinement/interp_conv/conv_1/Conv",
            # "/get_initdisp/GEMM_split1",
            # "/get_initdisp/GEMM_split0",
            # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0_1/Conv_split1",
            # "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0_1/Conv_split0",
            # "/backbone/mod2/mod2.0/head_layer/conv/conv.1/conv.1.0/Conv",
            # "/backbone/mod3/mod3.0/head_layer/downsample/downsample.0/Conv",
            # "/backbone/mod3/mod3.0/head_layer/conv/conv.0/conv.0.0/Conv",



        ]:
            for ii in range(len(node.inputs)):
                input_calib = find_input_calibration(node, ii)
                if input_calib and input_calib.tensor_type == "feature":
                    input_calib.qtype = "int8"
            # # 大部分节点都是单输出的，所以转换在输出类型上不支持配置指定idx的输出数据类型.
            # output_calib = find_output_calibration(node)
            # # 这里一般不需要再判断tensor_type为feature类型,因为weight HzCalibration只在Conv的输入出现
            # if output_calib:
            #     output_calib.qtype = "int8"
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
    exp_root = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v2/"
    onnx_prefix = 'PTQ_check_yuv444'
    validate_instereo2k(exp_root, onnx_prefix, left, right, disp_gt)
