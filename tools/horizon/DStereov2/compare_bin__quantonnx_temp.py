import os
import cv2
import copy
import numpy as np
import os.path as osp
from PIL import Image
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

infra1_nv12_path = "/root/temp/left_x5_nv12.bin"
infra2_nv12_path = "/root/temp/right_x5_nv12.bin"

nv12_data1 = np.fromfile(infra1_nv12_path, dtype=np.int8)
nv12_data2 = np.fromfile(infra2_nv12_path, dtype=np.int8)

yuv444_data1 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data1)
yuv444_data1 = np.transpose(yuv444_data1, (1, 2, 0))
save_yuv444_data1 = np.fromfile("/root/temp/left_x5_yuv444.bin", dtype=np.uint8).reshape(352, 640,3)
assert np.max(np.abs(yuv444_data1 - save_yuv444_data1)) == 0, f"yuv444_data1 diff max: {np.max(np.abs(yuv444_data1 - save_yuv444_data1))}"
yuv444_data2 = NV12ToYUV444Transformer((352, 640), yuv444_output_layout="CHW").run_transform(nv12_data2)
yuv444_data2 = np.transpose(yuv444_data2, (1, 2, 0))
save_yuv444_data2 = np.fromfile("/root/temp/right_x5_yuv444.bin", dtype=np.uint8).reshape(352, 640,3)
assert np.max(np.abs(yuv444_data2 - save_yuv444_data2)) == 0, f"yuv444_data2 diff max: {np.max(np.abs(yuv444_data2 - save_yuv444_data2))}"

yuv444_128_left = yuv444_data1[np.newaxis, ...]
yuv444_128_left = yuv444_128_left - 128
yuv444_128_left = yuv444_128_left.astype(np.int8)
yuv444_128_left = np.ascontiguousarray(yuv444_128_left)
save_yuv444_128_left = np.fromfile("/root/temp/left_x5_yuv444_128.npy", dtype=np.int8).reshape(1, 352, 640,3)
assert np.max(np.abs(yuv444_128_left - save_yuv444_128_left)) == 0, f"yuv444_128_left diff max: {np.max(np.abs(yuv444_128_left - save_yuv444_128_left))}"

yuv444_128_right = yuv444_data2[np.newaxis, ...]
yuv444_128_right = yuv444_128_right - 128
yuv444_128_right = yuv444_128_right.astype(np.int8)
yuv444_128_right = np.ascontiguousarray(yuv444_128_right)
save_yuv444_128_right = np.fromfile("/root/temp/right_x5_yuv444_128.npy", dtype=np.int8).reshape(1, 352, 640,3)
assert np.max(np.abs(yuv444_128_right - save_yuv444_128_right)) == 0, f"yuv444_128_right diff max: {np.max(np.abs(yuv444_128_right - save_yuv444_128_right))}"

onnx_model_path = "/root/temp/PTQ_check_yuv444_quantized_model.onnx"
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

# -----------BIN hrt_model_exec infer --model_file PTQ_check_yuv444_optimized_float_model_split.bin --input_file left_x5_nv12.bin,right_x5_nv12.bin  --enable_dump --dump_format bin
# -----------BIN hrt_model_exec infer --model_file PTQ_check_yuv444.bin --input_file left_x5_nv12.bin,right_x5_nv12.bin  --enable_dump --dump_format bin
board_disp = np.fromfile("/root/temp/model_infer_output_0_disp.bin", dtype=np.float32)
board_disp = np.reshape(board_disp, (1,9, 352,640))
# board_disp = np.transpose(board_disp, (0, 3, 1, 2))
board_disp = np.ascontiguousarray(board_disp)
board_disp = board_disp.astype(np.float32)
# board_disp *= 0.005985679570585489

board_spx = np.fromfile("/root/temp/model_infer_output_1_spx.bin", dtype=np.float32)
board_spx = np.reshape(board_spx, (1,9,352,640))
# board_spx = np.transpose(board_spx, (0, 3, 1, 2))
board_spx = board_spx.astype(np.float32)
# board_spx *= 0.000030518509447574615

print(f"spx diff: {np.max(np.abs(board_spx - spx))}")
print(f"disp diff: {np.max(np.abs(board_disp - disp))}")
