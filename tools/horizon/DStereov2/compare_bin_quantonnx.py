import os
import cv2
import copy
import numpy as np
import os.path as osp
from PIL import Image
from horizon_tc_ui import HB_ONNXRuntime


if __name__ == "__main__":
    output_disp_npy = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/output_v1_demo_result/model_infer_output_0_disp.bin"
    output_spx_npy = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/output_v1_demo_result/model_infer_output_1_spx.bin"
    # onnx_disp_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/disp_unfold.npy"
    # onnx_spx_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/spx.npy"

    infra1_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/left_x5_yuv444_128.npy"
    infra2_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1_demo_result/right_x5_yuv444_128.npy"
    onnx_model_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v1/PTQ_check_yuv444_quantized_model.onnx"


    # quant onnx输出
    infra1 = np.fromfile(infra1_path, dtype=np.int8).reshape(1, 352, 640, 3)
    infra2 = np.fromfile(infra2_path, dtype=np.int8).reshape(1, 352, 640, 3)

    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input_.name for input_ in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    feed_dict = {
            input_names[0]: infra1,
            input_names[1]: infra2,
        }
    disp_unfold, spx = sess.run(output_names[:2], feed_dict)

    # bin 输出
    disp_bin = np.fromfile(output_disp_npy, dtype=np.float32).reshape(1, 9, 352, 640)
    disp_bin = disp_bin * 0.00004713133239420131
    spx_bin = np.fromfile(output_spx_npy, dtype=np.int16).reshape(1, 9, 352, 640)
    spx_bin = spx_bin * 0.000030518509447574615

    print(f"disp diff max: {np.max(np.abs(disp_unfold - disp_bin))}")
    print(f"spx diff max: {np.max(np.abs(spx - spx_bin))}")
   