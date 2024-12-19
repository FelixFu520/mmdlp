from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *
import os
import cv2
import sys
import numpy as np
import torch

if __name__ == "__main__":
    infra1_nv12_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/left_x5_nv12.bin"
    infra2_nv12_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/infer_result/right_x5_nv12.bin"

    nv12_data1 = np.fromfile(infra1_nv12_path, dtype=np.int8)
    nv12_data2 = np.fromfile(infra2_nv12_path, dtype=np.int8)

    yuv444_data1 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data1)
    yuv444_data2 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data2)

    yuv444_128_left = yuv444_data1[np.newaxis, ...]
    yuv444_128_left = yuv444_128_left - 128
    yuv444_128_left = yuv444_128_left.astype(np.int8)
    yuv444_128_left = yuv444_128_left.transpose(0, 2, 3, 1)

    yuv444_128_right = yuv444_data2[np.newaxis, ...]
    yuv444_128_right = yuv444_128_right - 128
    yuv444_128_right = yuv444_128_right.astype(np.int8)
    yuv444_128_right = yuv444_128_right.transpose(0, 2, 3, 1)

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