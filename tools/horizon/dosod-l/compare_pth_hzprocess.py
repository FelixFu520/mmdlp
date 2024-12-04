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


def yuv444_128_to_rgb2(yuv_image):
    yuv_image = yuv_image.transpose(1, 2, 0)
    
    # 提取Y、U、V分量(这里假设数据排列顺序是Y、U、V通道依排列
    y_128 = yuv_image[:, :, 0].astype(np.float32)
    u_128 = yuv_image[:, :, 1].astype(np.float32)
    v_128 = yuv_image[:, :, 2].astype(np.float32)

    # YUV_BT601_FULL_RANGE_128 to YUV_BT601_FULL_RANGE.
    y = y_128 + 128
    u = u_128 + 128
    v = v_128 + 128

    # YUV_BT601_FULL_RANGE to RBG.
    r = 1 * y + 1.4017 * (v - 128)
    g = y - 0.3437 * (u - 128) - 0.7142 * (v - 128)
    b = y + 1.7722 * (u - 128)

    # data scale
    r = r / 255
    g = g / 255
    b = b / 255

    # YUV_BT601_FULL_RANGE_128 -> RGB -> data scale
    # r = (1 * (y_128 + 128) + 1.4017 * v_128) / 255
    # g = (1 * (y_128 + 128) - 0.3437 * u_128 - 0.7142 * v_128) / 255
    # b = (1 * (y_128 + 128) + 1.7722 * v_128) / 255

    # =====>
    # r = 0.003921568859368563 * y_128 + 0.005496862745098039 * v_128 + 0.5019607543945312
    # g = 0.003921568859368563 * y_128 - 0.0013478432083502412 * u_128 - 0.0028007845394313335 * v_128 + 0.5019607543945312
    # b = 0.003921568859368563 * y_128 + 0.0069498042576014996 * u_128 + 0.5019608736038208

    rgb_image = np.stack([r, g, b], axis=2) * 255
    # rgb_image = rgb_image.clip(0, 255).astype(np.uint8)
    rgb_image = rgb_image.transpose(2, 0, 1)
    return rgb_image



def yuv444_128_to_rgb(yuv_image):
    """
    将YUV444_128格式的图像数据转换为RGB格式
    """
    yuv_image = yuv_image.transpose(1, 2, 0)

    # 提取Y、U、V分量（这里假设数据排列顺序是Y、U、V通道依次排列）
    y = yuv_image[:, :, 0].astype(np.float32)
    u = yuv_image[:, :, 1].astype(np.float32)
    v = yuv_image[:, :, 2].astype(np.float32)

    r = 0.003921568859368563 * y + 0.005496863275766373 * v + 0.501960813999176
    g = 0.003921568859368563 * y - 0.0013478432083502412 * u - 0.0028007845394313335 * v + 0.5019607543945312
    b = 0.003921568859368563 * y + 0.0069498042576014996 * u + 0.5019608736038208

    # 合并RGB通道
    rgb_image = np.stack([r, g, b], axis=2) * 255
    # rgb_image = rgb_image.clip(0, 255).astype(np.uint8)
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
    fun_t3 = AddTransformer(-128)
    image_yuv444_128 = fun_t3.run_transform(image_yuv444)
    image_rgb = yuv444_128_to_rgb(image_yuv444_128).astype(np.float32)
    # image_rgb = yuv444_128_to_rgb2(image_yuv444_128).astype(np.float32)

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

    print(f"sum: {np.sum(np.abs(output - image_rgb))}")
    print(f"mean: {np.mean(output - image_rgb)}")
    print(f"max: {np.max(output - image_rgb)}")
    print(f"min: {np.min(output - image_rgb)}")

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
   
