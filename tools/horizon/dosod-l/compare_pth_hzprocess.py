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

def yuv444_128_to_rgb(yuv_image):
    """
    将YUV444_128格式的图像数据转换为RGB格式
    """
    yuv_image = yuv_image.transpose(1, 2, 0)

    # 提取Y、U、V分量（这里假设数据排列顺序是Y、U、V通道依次排列）
    y = yuv_image[:, :, 0].astype(np.float32)
    u = yuv_image[:, :, 1].astype(np.float32)
    v = yuv_image[:, :, 2].astype(np.float32)

    # r = (y + 128) +  0.005496863275766373 * v
    # g = (y + 128) -  0.0013478432083502412 * u - 0.0028007845394313335 * v
    # b = (y + 128) +  0.0069498042576014996 * u

    r = y +  0.005496863275766373 * v
    g = y -  0.0013478432083502412 * u - 0.0028007845394313335 * v
    b = y +  0.0069498042576014996 * u


    # 合并RGB通道
    rgb_image = np.stack([r, g, b], axis=2)
    rgb_image = rgb_image.clip(0, 255).astype(np.uint8)
    rgb_image = rgb_image.transpose(2, 0, 1)
    return rgb_image


def compare_rgb(image_path, onnx_origin_path, height, width):
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
    image_rgb = yuv444_128_to_rgb(image_yuv444).astype(np.float32)
    image_rgb /= 255

    # 得到HzPreprocess输出
    sess = HB_ONNXRuntime(model_file=onnx_origin_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    input_data = image_yuv444[np.newaxis, ...]
    feed_dict = {
        input_names[0]: input_data,
    }
    outputs = sess.run(output_names, feed_dict)
    output = outputs[0][0]

    diff = np.sum(np.abs(output - image_rgb))
    print(diff)

def compare_yuv444(image_path, onnx_origin_path, height, width):
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
    image_yuv = image_yuv444 / 255

    # 得到HzPreprocess输出
    sess = HB_ONNXRuntime(model_file=onnx_origin_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    input_data = image_yuv444[np.newaxis, ...]
    feed_dict = {
        input_names[0]: input_data,
    }
    outputs = sess.run(output_names, feed_dict)
    output = outputs[0][0]

    print(f"sum: {np.sum(np.abs(output - image_yuv))}")
    print(f"mean: {np.mean(output - image_yuv)}")
    print(f"max: {np.max(output - image_yuv)}")
    print(f"min: {np.min(output - image_yuv)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg")
    parser.add_argument('--onnx_origin_path', type=str, default="/home/fa.fu/work/work_dirs/horizon/dosod-l/20241203/output_v1/modified_dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_original_float_model.onnx")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")
    parser.add_argument("--mode", type=str, default="yuv444")
    args = parser.parse_args()

    image_path = args.image_path
    onnx_origin_path = args.onnx_origin_path
    height = args.height
    width = args.width

    if args.mode == "rgb":
        compare_rgb(image_path, onnx_origin_path, height, width)
    elif args.mode == "yuv444":
        compare_yuv444(image_path, onnx_origin_path, height, width)
   
