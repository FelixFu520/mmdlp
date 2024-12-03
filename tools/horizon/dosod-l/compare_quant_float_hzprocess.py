import os
import cv2
import numpy as np
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *
import onnx
from onnx import helper, checker, shape_inference
import argparse
from preprocess import preprocess_custom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg")
    parser.add_argument('--onnx_origin_path', type=str, default="/home/fa.fu/work/work_dirs/horizon/modified_dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_original_float_model.onnx")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")
    args = parser.parse_args()

    image_path = args.image_path
    onnx_origin_path = args.onnx_origin_path
    height = args.height
    width = args.width

    # 得到yuv444数据
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
    )
    image = image * 255
    fun_t1 = RGB2NV12Transformer(data_format="CHW")
    image_nv12 = fun_t1.run_transform(image)
    fun_t2 = NV12ToYUV444Transformer((height, width), yuv444_output_layout="CHW")
    image_yuv444 = fun_t2.run_transform(image_nv12)

    # pth输入
    fun_t3 = YUVTransformer("RGB")
    image_rgb = fun_t3.run_transform(image_yuv444)

    input_data = image_yuv444[np.newaxis, ...]


    input_data = input_data[np.newaxis, ...]
    input_data -= 128
    input_data = input_data.astype(np.int8)
    input_data = input_data.transpose(0, 2, 3, 1)

    # 得到HzPreprocess输出
    sess = HB_ONNXRuntime(model_file=onnx_origin_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
    )
    image = image * 255
    image = np.expand_dims(image, axis=0)
    fun_t = RGB2YUV444Transformer(data_format="CHW")
    input_data = fun_t.run_transform(image[0])
    input_data = input_data[np.newaxis, ...]
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)
    output = outputs[0][0]

    diff = np.sum(np.abs(output - image_pth))
    print(diff)
