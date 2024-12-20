import os
import cv2
import copy
import numpy as np
import os.path as osp
from PIL import Image
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *


def test1():
    output_disp_npy = "/root/model_infer_output_0_disp.bin"
    output_spx_npy = "/root/model_infer_output_1_spx.bin"
    onnx_disp_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/disp_unfold.npy"
    onnx_spx_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/spx.npy"

    # quant onnx输出
    disp = np.load(onnx_disp_path)
    spx = np.load(onnx_spx_path)

    # bin 输出
    disp_bin = np.fromfile(output_disp_npy, dtype=np.int16).reshape(1, 9, 352, 640)
    spx_bin = np.fromfile(output_spx_npy, dtype=np.int16).reshape(1, 9, 352, 640)

    print(f"disp diff max: {np.max(np.abs(disp - disp_bin))}")
    print(f"spx diff max: {np.max(np.abs(spx - spx_bin))}")


def test2():
    output_disp_npy = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/model_infer_output_0_disp.bin"
    output_spx_npy = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/model_infer_output_1_spx.bin"
    onnx_disp_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/left_x5_yuv444_128.npy"
    onnx_spx_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/right_x5_yuv444_128.npy"


    # quant onnx输出
    infra1 = np.fromfile(onnx_disp_path, dtype=np.int8).reshape(1, 352, 640, 3)
    infra2 = np.fromfile(onnx_spx_path, dtype=np.int8).reshape(1, 352, 640, 3)

    sess = HB_ONNXRuntime(model_file="/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/PTQ_check_yuv444_quantized_model.onnx")
    input_names = [input_.name for input_ in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    feed_dict = {
            input_names[0]: infra1,
            input_names[1]: infra2,
        }
    disp_unfold, spx = sess.run(output_names, feed_dict)

    # bin 输出
    disp_bin = np.fromfile(output_disp_npy, dtype=np.float32).reshape(1, 9, 352, 640)
    spx_bin = np.fromfile(output_spx_npy, dtype=np.float32).reshape(1, 9, 352, 640)

    print(f"disp diff max: {np.max(np.abs(disp_unfold - disp_bin))}")
    print(f"spx diff max: {np.max(np.abs(spx - spx_bin))}")


def test3():
    infra1_nv12_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/left_x5_nv12.bin"
    infra2_nv12_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/right_x5_nv12.bin"

    nv12_data1 = np.fromfile(infra1_nv12_path, dtype=np.int8)
    nv12_data2 = np.fromfile(infra2_nv12_path, dtype=np.int8)

    yuv444_data1 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data1)
    yuv444_data1 = np.transpose(yuv444_data1, (1, 2, 0))
    save_yuv444_data1 = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/left_x5_yuv444.bin", dtype=np.uint8).reshape(352, 640,3)
    assert np.max(np.abs(yuv444_data1 - save_yuv444_data1)) == 0, f"yuv444_data1 diff max: {np.max(np.abs(yuv444_data1 - save_yuv444_data1))}"
    yuv444_data2 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data2)
    yuv444_data2 = np.transpose(yuv444_data2, (1, 2, 0))
    save_yuv444_data2 = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/right_x5_yuv444.bin", dtype=np.uint8).reshape(352, 640,3)
    assert np.max(np.abs(yuv444_data2 - save_yuv444_data2)) == 0, f"yuv444_data2 diff max: {np.max(np.abs(yuv444_data2 - save_yuv444_data2))}"

    yuv444_128_left = yuv444_data1[np.newaxis, ...]
    yuv444_128_left = yuv444_128_left - 128
    yuv444_128_left = yuv444_128_left.astype(np.int8)
    save_yuv444_128_left = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/left_x5_yuv444_128.npy", dtype=np.int8).reshape(1, 352, 640,3)
    assert np.max(np.abs(yuv444_128_left - save_yuv444_128_left)) == 0, f"yuv444_128_left diff max: {np.max(np.abs(yuv444_128_left - save_yuv444_128_left))}"

    yuv444_128_right = yuv444_data2[np.newaxis, ...]
    yuv444_128_right = yuv444_128_right - 128
    yuv444_128_right = yuv444_128_right.astype(np.int8)
    save_yuv444_128_right = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/right_x5_yuv444_128.npy", dtype=np.int8).reshape(1, 352, 640,3)
    assert np.max(np.abs(yuv444_128_right - save_yuv444_128_right)) == 0, f"yuv444_128_right diff max: {np.max(np.abs(yuv444_128_right - save_yuv444_128_right))}"

    onnx_model_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/PTQ_check_yuv444_quantized_model.onnx"
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    feed_dict = {
        input_names[0]: yuv444_128_left,
        input_names[1]: yuv444_128_right,
    }

    outputs = sess.run(output_names, feed_dict)

    disp = outputs[0]
    spx = outputs[1]

    # -----------BIN
    board_disp = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/model_infer_output_0_disp.bin", dtype=np.float32)
    board_disp = np.reshape(board_disp, (1,9,352,640))

    board_spx = np.fromfile("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/model_infer_output_1_spx.bin", dtype=np.float32)
    board_spx = np.reshape(board_spx, (1,9,352,640))

    print(f"spx diff: {np.max(np.abs(board_spx - spx))}")
    print(f"disp diff: {np.max(np.abs(board_disp - disp))}")


if __name__ == "__main__":
    print("----------------直接从npy中读取quant onnx推理结果----------------")
    test1()
    # print("----------------使用quant onnx 重新推理, 使用evaluate_ptq.py脚本中的前处理----------------")
    # test2()
    # print("----------------使用quant onnx 重新推理, 使用重新写的前处理----------------")
    # test3()