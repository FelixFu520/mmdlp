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


def infer_quant_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_image(
        image_path, 
        height=height, 
        width=width,
        bgr_to_rgb=True,
        to_float=True,
        mean_std=False, 
        transpose=True, 
        new_axis=False
    )
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

    if result_dir is not None:
        # NMS
        scores, bboxes = outputs
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.6, 0.5)

        # 画图
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)


if __name__ == "__main__":
    onnx_model_path = "/home/users/fa.fu/work/work_dirs/dosod/20241103/output/DOSOD_L_without_nms_v0.1_quantized_model.onnx"
    image_path = "/home/users/fa.fu/work/work_dirs/dosod/demo_images/030125.jpg"
    result_dir = "/home/users/fa.fu/work/work_dirs/dosod/result"
    os.makedirs(result_dir, exist_ok=True)

    # 使用原始onnx推理查看下onnx是否正确
    infer_quant_onnx(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
        height=640,
        width=640
    )