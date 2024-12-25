import numpy as np
import argparse
import cv2
from typing import Tuple, Union
from mmcv.transforms import LoadAnnotations, LoadImageFromFile
from mmyolo.datasets.transforms import YOLOv5KeepRatioResize, LetterResize
from prepcocess import preprocess_image
MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

def get_rescale_ratio(old_size: Tuple[int, int],
                        scale: Union[float, Tuple[int]]) -> float:
    """Calculate the ratio for rescaling.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by
            this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within
            the scale.

    Returns:
        float: The resize ratio.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                            max_short_edge / min(h, w))
    else:
        raise TypeError('Scale must be a number or tuple of int, '
                        f'but got {type(scale)}')

    return scale_factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare input")
    parser.add_argument("--img_path", type=str, default="/home/fa.fu/work/work_dirs/dosod_debug/demo_images/0892.jpg")
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=896)
    args = parser.parse_args()

    img_path = args.img_path
    height = args.height
    width = args.width


    # 准备pth输入的数据, 从训练的配置文件中抄过来的
    img_path_dict = {"img_path": img_path}
    imgs = LoadImageFromFile().transform(img_path_dict)
    img_scale = (width, height)
    imgs = YOLOv5KeepRatioResize(scale=img_scale).transform(imgs)
    imgs = LetterResize(scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)).transform(imgs)
    input_data1 = imgs["img"]
    input_data1 = input_data1.astype(np.float32) 
    input_data1 = input_data1[..., ::-1]
    input_data1 /= 255
    input_data1 = input_data1.transpose(2, 0, 1)


    # 准备onnx输出的数据
    img = cv2.imread(img_path)
    original_h, original_w = img.shape[:2]
    ratio = get_rescale_ratio((original_h, original_w), img_scale)
    print("ratio:", ratio)
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    img = cv2.resize(
        img, (int(original_w * ratio), int(original_h * ratio)),
        interpolation=interpolation)
    new_height, new_width = img.shape[:2]
    top_pad = (height - new_height) // 2
    bottom_pad = height - new_height - top_pad
    left_pad = (width - new_width) // 2
    right_pad = width - new_width - left_pad
    img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # LetterResize  这里面的相当于没有触发到
    input_data2 = img.astype(np.float32) 
    # BGR TO RGB 
    input_data2 = input_data2[..., ::-1]
    input_data2 /= 255
    input_data2 = input_data2.transpose(2, 0, 1)
    
    # 对比
    max_diff = np.abs(input_data1 - input_data2).max()
    print("max_diff:", max_diff)
