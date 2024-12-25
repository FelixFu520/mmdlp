import os
import cv2
import copy
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import os.path as osp
from matplotlib.figure import Figure
import onnxruntime as ort
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

from prepcocess import preprocess_image
from postprocess import postprocess


def infer_quant_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_image(image_path, mean_std=False, to_float=True, 
                             bgr_to_rgb=True, transpose=True, new_axis=False, 
                             height=height, width=width)
    fun_t = RGB2YUV444Transformer(data_format="CHW")
    input_data = fun_t.run_transform(image)
    input_data = input_data[np.newaxis, ...]
    input_data -= 128
    input_data = input_data.astype(np.int8)
    input_data = input_data.transpose(0, 2, 3, 1)
    # infer
    feed_dict = {
        input_names[0]: input_data,
    }
    outputs = sess.run(output_names, feed_dict)

    # postprocess
    postprocess(outputs, width=width, height=height, result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]), image_path=image_path)


if __name__ == "__main__":
    onnx_model_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/output-stdc1_pre_1024×2048_jj/stdc1_pre_1024×2048_jj_quantized_model.onnx"
    image_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/demo_images/krefeld_000000_012353_leftImg8bit.png"
    result_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/result"
    os.makedirs(result_dir, exist_ok=True)

    # 使用原始onnx推理查看下onnx是否正确
    infer_quant_onnx(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
        height=1024,
        width=2048
    )