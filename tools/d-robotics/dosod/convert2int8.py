
import os
import os.path as osp
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from horizon_nn.ir import load_model, save_model
from horizon_nn.common import find_input_calibration, find_output_calibration
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *
from preprocess import preprocess_custom_v2
from find_node_to_int8 import get_onnx_nodes

def is_bad(score_ori, score_modify, threshold):
    return abs(score_ori - score_modify) > threshold

# def can_be_skipped(node: OnnxNode) -> bool:
#     """判断向前或者向后查找校准节点的时候, 遇到的节点是否可以跳过."""
#     if node.op_type in [
#         "MaxPool",
#         "GlobalMaxPool",
#         "Relu",
#         "Reshape",
#         "Transpose",
#         "ReduceMax",
#         "Split",
#         "Slice",
#         "Gather",
#         "ScatterND",
#     ]:
#         return True

#     return False
# def find_input_calibration(
#     node: OnnxNode,
#     index: Union[None, int] = None,
# ) -> Optional[CalibrationNode]:
#     """找到一个普通节点输入的校准节点.

#     Args:
#         node: 普通非校准节点, 用于对其输入寻找校准节点.
#         index: 如果是None, 依次遍历节点各个输入, 返回找到的第一个校准节点;
#                如果不是None, 返回指定输入对应的校准节点.

#     Returns:
#         如果找到返回相应校准节点, 否则返回None.
#     """
#     if len(node.prev_ops) > 0:
#         if index is None and node.op_type in ["ScatterND"]:
#             # ScatterND节点的输入0是data, 如果没有明确指定index, 默认查找data输入上的校准节点.
#             index = 0
#         if index is None:
#             for prev in node.prev_ops:
#                 if prev.op_type == "HzCalibration":
#                     return prev
#             for prev in node.prev_ops:
#                 if can_be_skipped(prev):
#                     return find_input_calibration(prev)
#         elif node.inputs[index].src_op is not None:
#             prev = node.inputs[index].src_op
#             if prev.op_type == "HzCalibration":
#                 return prev
#             if can_be_skipped(prev):
#                 return find_input_calibration(prev)

#     return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model to int8')
    parser.add_argument('--calib_onnx_path', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/output_v4/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444_calibrated_model.onnx', help='Model path')
    parser.add_argument('--output_path', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/output_v4/', help='Output file path')
    parser.add_argument('--percent', type=float, default=0.001, help='percent')
    parser.add_argument('--image_path', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/demo_images/0892.jpg', help='Image path')
    parser.add_argument('--height', type=int, default=672, help='Height')
    parser.add_argument('--width', type=int, default=896, help='Width')

    args = parser.parse_args()


    # ----------image
    image = preprocess_custom_v2(args.image_path, args.height, args.width)
    image = image * 255
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # ----------Calib, 获得score
    sess = HB_ONNXRuntime(model_file=args.calib_onnx_path)
    input_names  = [input.name for input in sess.get_inputs()]
    output_names  = [output.name for output in sess.get_outputs()]
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)
    scores, bboxes = outputs
    bboxes = bboxes.squeeze(0)
    scores = scores.squeeze(0)
    argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
    argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
    indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.3, 0.5)
    assert len(indexs) == 1
    score = argmax_scores[indexs].item()


    node_dict = get_onnx_nodes(args.calib_onnx_path)


    # ----------遍历所有Conv节点
    node_int8_ok = []
    node_int8_bad = []

    node_conv_set = set(node_dict['Conv'])
    for node_name in node_conv_set:
        print("----------------------------------------")
        # 加载权重, 修改conv, 保存
        calibrated_model = load_model(args.calib_onnx_path)
        for node in calibrated_model.graph.nodes:
            if node.name in set(node_int8_ok + [node_name]):
                for ii in range(len(node.inputs)):
                    input_calib = find_input_calibration(node, ii)
                    if input_calib and input_calib.tensor_type == "feature":
                        input_calib.qtype = "int8"
        temp_path = os.path.join(args.output_path, 'temp.onnx')
        save_model(calibrated_model, temp_path)
        for n in set(node_int8_ok + [node_name]):
            print(n)
        print("----")

        # 使用修改后的模型进行推理
        sess_calib = HB_ONNXRuntime(model_file=temp_path)
        input_names_calib  = [input.name for input in sess_calib.get_inputs()]
        output_names_calib  = [output.name for output in sess_calib.get_outputs()]
        feed_dict = {
            input_names_calib [0]: image,
        }
        outputs_calib = sess_calib.run(output_names_calib, feed_dict)
        scores_calib, bboxes_calib = outputs_calib
        bboxes_calib = bboxes_calib.squeeze(0)
        scores_calib = scores_calib.squeeze(0)
        argmax_idx_calib = np.argmax(scores_calib, axis=1).astype(np.int8)
        argmax_scores_calib = scores_calib[np.arange(scores_calib.shape[0]), argmax_idx_calib]
        indexs_calib = cv2.dnn.NMSBoxes(bboxes_calib, argmax_scores_calib, 0.3, 0.5)
        assert len(indexs_calib) == 1
        score_calib = argmax_scores_calib[indexs_calib].item()

        if is_bad(score, score_calib, args.percent):
            print(f'bad:{node_name}, score:{score}, score_calib:{score_calib}, abs(score - score_calib):{abs(score - score_calib)}')
            node_int8_bad.append(node_name)
        else:
            print(f'ok:{node_name}, score:{score}, score_calib:{score_calib}, abs(score - score_calib):{abs(score - score_calib)}')
            node_int8_ok.append(node_name)
        

        print("----------------------------------------")

    print('node_conv_int8_ok:', node_int8_ok)
    print('node_conv_int8_bad:', node_int8_bad)

    # ----------遍历所有Other节点
    node_other_int8_ok = []
    node_other_int8_bad = []
    node_other_set = set(node_dict['Other'])
    for node_name in node_other_set:
        print("----------------------------------------")
        # 加载权重, 修改conv, 保存
        calibrated_model = load_model(args.calib_onnx_path)
        for node in calibrated_model.graph.nodes:
            if node.name in set(node_int8_ok + node_other_int8_ok + [node_name]):
                for ii in range(len(node.inputs)):
                    input_calib = find_input_calibration(node, ii)
                    if input_calib and input_calib.tensor_type == "feature":
                        input_calib.qtype = "int8"
        temp_path = os.path.join(args.output_path, 'temp.onnx')
        save_model(calibrated_model, temp_path)
        for n in set(node_int8_ok + node_other_int8_ok + [node_name]):
            print(n)
        print("----")

        # 使用修改后的模型进行推理
        sess_calib = HB_ONNXRuntime(model_file=temp_path)
        input_names_calib  = [input.name for input in sess_calib.get_inputs()]
        output_names_calib  = [output.name for output in sess_calib.get_outputs()]
        feed_dict = {
            input_names_calib [0]: image,
        }
        outputs_calib = sess_calib.run(output_names_calib, feed_dict)
        scores_calib, bboxes_calib = outputs_calib
        bboxes_calib = bboxes_calib.squeeze(0)
        scores_calib = scores_calib.squeeze(0)
        argmax_idx_calib = np.argmax(scores_calib, axis=1).astype(np.int8)
        argmax_scores_calib = scores_calib[np.arange(scores_calib.shape[0]), argmax_idx_calib]
        indexs_calib = cv2.dnn.NMSBoxes(bboxes_calib, argmax_scores_calib, 0.3, 0.5)
        assert len(indexs_calib) == 1
        score_calib = argmax_scores_calib[indexs_calib].item()

        if is_bad(score, score_calib, args.percent):
            print(f'bad:{node_name}, score:{score}, score_calib:{score_calib}, abs(score - score_calib):{abs(score - score_calib)}')
            node_other_int8_bad.append(node_name)
        else:
            print(f'ok:{node_name}, score:{score}, score_calib:{score_calib}, abs(score - score_calib):{abs(score - score_calib)}')
            node_other_int8_ok.append(node_name)
        
        print("----------------------------------------")


    print('node_int8_ok:', node_other_int8_ok)
    print('node_int8_bad:', node_other_int8_bad)

    # ----------保存结果
    with open(os.path.join(args.output_path, 'node_int8_ok.txt'), 'w') as f:
        for node in node_int8_ok:
            f.write("Conv-" + node + '\n')
        for node in node_other_int8_ok:
            f.write("Others-" + node + '\n')
    with open(os.path.join(args.output_path, 'node_int8_bad.txt'), 'w') as f:
        for node in node_int8_bad:
            f.write("Conv-" + node + '\n')
        for node in node_other_int8_bad:
            f.write("Others-" + node + '\n')