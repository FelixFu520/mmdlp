import argparse
import cv2
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import Tuple, Union
from mmcv.transforms import LoadAnnotations, LoadImageFromFile
from mmyolo.datasets.transforms import YOLOv5KeepRatioResize, LetterResize
from utils import BGR2NV12Transformer, NV12ToYUV444Transformer

__all__ = ['preprocess_custom_v1', 'preprocess_custom_v2', 'preprocess_mmcv_v1', 'preprocess_mmcv_v2']

def _get_rescale_ratio(old_size: Tuple[int, int],
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

def mergeUV(u, v):
    if u.shape == v.shape:
        uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
        for i in range(0, u.shape[0]):
            for j in range(0, u.shape[1]):
                uv[i, 2 * j] = u[i, j]
                uv[i, 2 * j + 1] = v[i, j]
        return uv
    else:
        raise ValueError("size of Channel U is different with Channel V")

def rgb2nv12_calc(image):
    if image.ndim == 3:
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        y = (0.299 * r + 0.587 * g + 0.114 * b)
        u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
        v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
        uv = mergeUV(u, v)
        yuv = np.vstack((y, uv))
        return yuv.astype(np.uint8)

def bgr2nv12(image) -> np.ndarray:
    # image HWC
    image_shape = image.shape[:-1]
    if image_shape[0] * image_shape[1] % 2 != 0:
        raise ValueError(
            f"Invalid odd shape: {image_shape[0]} x {image_shape[1]}, "
            "expect even number for height and width")

    image = rgb2nv12_calc(image)
    return image

def nv12Toyuv444(image, height, width, yuv444_output_layout='HWC') -> dict:
    nv12_image = image.flatten()
    yuv444 = np.empty([height, width, 3], dtype=np.uint8)
    yuv444[:, :, 0] = nv12_image[:width * height].reshape(
        height, width)
    u = nv12_image[width * height::2].reshape(
        height // 2, width // 2)
    yuv444[:, :, 1] = Image.fromarray(u).resize((width, height),
                                                resample=0)
    v = nv12_image[width * height + 1::2].reshape(
        height // 2, width // 2)
    yuv444[:, :, 2] = Image.fromarray(v).resize((width, height),
                                                resample=0)
    image = yuv444.astype(np.uint8)
    if yuv444_output_layout == "CHW":
        image = np.transpose(image, (2, 0, 1))
    return image




def preprocess_custom_v1(image_path: str, height: int = 672, width: int = 896, 
                      allow_scale_up: bool = False,
                      pad_val = 114) -> np.ndarray:
    """
    用于模拟preprocess_mmcv_v1

        主要使用mmcv库对图像进行前处理
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(896, 672,), type='YOLOv5KeepRatioResize'),
        dict(allow_scale_up=False, pad_val=dict(img=114), scale=(896, 672,), type='LetterResize'),

        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                0.0,
                0.0,
                0.0,
            ],
            std=[
                255.0,
                255.0,
                255.0,
            ],
        type='YOLOWDetDataPreprocessor')
    """

    # 代替LoadImageFromFile
    # 1.read
    image = cv2.imread(image_path)
    # gray2bgr
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    

    # 代替YOLOv5KeepRatioResize
    # 3. resize
    scale = (width, height)
    original_h, original_w = image.shape[:2]
    ratio = _get_rescale_ratio((original_h, original_w), scale)
    if ratio != 1:
        # resize image according to the shape
        # NOTE: We are currently testing on COCO that modifying
        # this code will not affect the results.
        # If you find that it has an effect on your results,
        # please feel free to contact us.
        image = cv2.resize(
            image, 
            (int(original_w * ratio), int(original_h * ratio)), 
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)


    # 代替LetterResize
    # 4. resize & pad
    scale = (width, height)
    scale = scale[::-1]  # wh -> hw

    image_shape = image.shape[:2]  # height, width
    # Scale ratio (new / old)
    ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not allow_scale_up:
        ratio = min(ratio, 1.0)

    ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

    # compute the best size of the image
    no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                    int(round(image_shape[1] * ratio[1])))

    # padding height & width
    padding_h, padding_w = [
        scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
    ]


    if image_shape != no_pad_shape:
        # compare with no resize and padding size
        image = cv2.resize(
            image, (no_pad_shape[1], no_pad_shape[0]), interpolation=cv2.INTER_LINEAR)

    # padding
    top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
        round(padding_w // 2 - 0.1))
    bottom_padding = padding_h - top_padding
    right_padding = padding_w - left_padding

    padding_list = [
        top_padding, bottom_padding, left_padding, right_padding
    ]
    if top_padding != 0 or bottom_padding != 0 or \
            left_padding != 0 or right_padding != 0:

        if isinstance(pad_val, int) and image.ndim == 3:
            pad_val = tuple(pad_val for _ in range(image.shape[2]))

        padding=(padding_list[2], padding_list[0], padding_list[3],
                                padding_list[1])
        image = cv2.copyMakeBorder(
            image,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2.BORDER_CONSTANT,
            value=pad_val)
            
    
    # 这一部分是model的data_preprocess
    input_data = image
    input_data = input_data.astype(np.float32)
    input_data = input_data[..., ::-1]
    input_data /= 255
    input_data = input_data.transpose(2, 0, 1)

    # 确保连续
    input_data = np.ascontiguousarray(input_data)  # 确保数据在内存中是连续的

    return input_data

def preprocess_custom_v2(image_path: str, height: int = 672, width: int = 896, 
                      allow_scale_up: bool = False,
                      pad_val = 114) -> np.ndarray:
    """
    用于模拟preprocess_mmcv_v2
    """

    # 代替LoadImageFromFile
    # 1.read
    image = cv2.imread(image_path)
    # gray2bgr
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    

    # 代替YOLOv5KeepRatioResize
    # 3. resize
    scale = (width, height)
    original_h, original_w = image.shape[:2]
    ratio = _get_rescale_ratio((original_h, original_w), scale)
    if ratio != 1:
        # resize image according to the shape
        # NOTE: We are currently testing on COCO that modifying
        # this code will not affect the results.
        # If you find that it has an effect on your results,
        # please feel free to contact us.
        image = cv2.resize(
            image, 
            (int(original_w * ratio), int(original_h * ratio)), 
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)


    # 代替LetterResize
    # 4. resize & pad
    scale = (width, height)
    scale = scale[::-1]  # wh -> hw

    image_shape = image.shape[:2]  # height, width
    # Scale ratio (new / old)
    ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not allow_scale_up:
        ratio = min(ratio, 1.0)

    ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

    # compute the best size of the image
    no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                    int(round(image_shape[1] * ratio[1])))

    # padding height & width
    padding_h, padding_w = [
        scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
    ]


    if image_shape != no_pad_shape:
        # compare with no resize and padding size
        image = cv2.resize(
            image, (no_pad_shape[1], no_pad_shape[0]), interpolation=cv2.INTER_LINEAR)

    # padding
    top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
        round(padding_w // 2 - 0.1))
    bottom_padding = padding_h - top_padding
    right_padding = padding_w - left_padding

    padding_list = [
        top_padding, bottom_padding, left_padding, right_padding
    ]
    if top_padding != 0 or bottom_padding != 0 or \
            left_padding != 0 or right_padding != 0:

        if isinstance(pad_val, int) and image.ndim == 3:
            pad_val = tuple(pad_val for _ in range(image.shape[2]))

        padding=(padding_list[2], padding_list[0], padding_list[3],
                                padding_list[1])
        image = cv2.copyMakeBorder(
            image,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2.BORDER_CONSTANT,
            value=pad_val)
            
    # # 代替BGR2NV12Transformer
    image = bgr2nv12(image)

    # # 代替NV12ToYUV444Transformer
    image = nv12Toyuv444(image, height, width, yuv444_output_layout="CHW")

    # # 这一部分是model的data_preprocess
    image = image.astype(np.float32)
    image /= 255

    # 确保连续
    image = np.ascontiguousarray(image)  # 确保数据在内存中是连续的

    return image

def preprocess_mmcv_v1(image_path: str, height: int = 672, width: int = 896) -> np.ndarray:
    """
    主要使用mmcv库对图像进行前处理
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(896, 672,), type='YOLOv5KeepRatioResize'),
        dict(allow_scale_up=False, pad_val=dict(img=114), scale=(896, 672,), type='LetterResize'),

        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                0.0,
                0.0,
                0.0,
            ],
            std=[
                255.0,
                255.0,
                255.0,
            ],
        type='YOLOWDetDataPreprocessor')
    """
    # 这一部分是pipeline中的
    image_path_dict = {"img_path": image_path}
    image = LoadImageFromFile().transform(image_path_dict)
    image = YOLOv5KeepRatioResize(scale=(width, height)).transform(image)
    image = LetterResize(scale=(width, height), allow_scale_up=False, pad_val=dict(img=114)).transform(image)

    # 这一部分是model的data_preprocess
    input_data = image['img']   # 'img' is the key of the image
    input_data = input_data.astype(np.float32)  # to_float
    input_data = input_data[..., ::-1]  # bgr2rgb
    input_data /= 255   # mean_std
    input_data = input_data.transpose(2, 0, 1)  # transpose to (C, H, W)

    assert input_data.shape == (3, height, width)

    return input_data

def preprocess_mmcv_v2(image_path: str, height: int = 672, width: int = 896) -> np.ndarray:
    """
    主要使用mmcv库对图像进行前处理
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(896, 672,), type='YOLOv5KeepRatioResize'),
        dict(allow_scale_up=False, pad_val=dict(img=114), scale=(896, 672,), type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(data_format='HWC', type='BGR2NV12Transformer'),
        dict(target_size=(672, 896), type='NV12ToYUV444Transformer', yuv444_output_layout='HWC'),
        dict(type='LoadText'),
        dict(meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ), type='mmdet.PackDetInputs'),
        
        data_preprocessor=dict(
            bgr_to_rgb=False,
            mean=[
                0.0,
                0.0,
                0.0,
            ],
            std=[
                255.0,
                255.0,
                255.0,
            ],
        type='YOLOWDetDataPreprocessor')
       
    """
    # 这一部分是pipeline中的
    image_path_dict = {"img_path": image_path}
    image = LoadImageFromFile().transform(image_path_dict)
    image = YOLOv5KeepRatioResize(scale=(width, height)).transform(image)
    image = LetterResize(scale=(width, height), allow_scale_up=False, pad_val=dict(img=114)).transform(image)
    image = BGR2NV12Transformer(data_format='HWC').transform(image)
    image = NV12ToYUV444Transformer(yuv444_output_layout="CHW", target_size=(height, width)).transform(image)


    # # 这一部分是model的data_preprocess
    image = image['img']
    image = image.astype(np.float32)
    image /= 255

    assert image.shape == (3, height, width)

    image = np.ascontiguousarray(image)  # 确保数据在内存中是连续的

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare input")
    parser.add_argument("--images_dir", type=str, default="/home/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103")
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--mode", type=str, default="v1")

    args = parser.parse_args()


    images_path = [os.path.join(args.images_dir, img) for img in os.listdir(args.images_dir) if img.endswith('.jpg')]

    if args.mode == "v1":
        for img_p in tqdm(images_path, desc="compare input"):
            image = preprocess_custom_v1(img_p, height=args.height, width=args.width)
            image_mmcv = preprocess_mmcv_v1(img_p, height=args.height, width=args.width)

            max_diff = np.sum(np.abs(image - image_mmcv))
            if max_diff > 0:
                print(f"{os.path.basename(img_p)} max_diff: {max_diff}")
    elif args.mode == "v2":
        for img_p in tqdm(images_path, desc="compare input"):
            image = preprocess_custom_v2(img_p, height=args.height, width=args.width)
            image_mmcv = preprocess_mmcv_v2(img_p, height=args.height, width=args.width)

            max_diff = np.sum(np.abs(image - image_mmcv))
            if max_diff > 0:
                print(f"{os.path.basename(img_p)} max_diff: {max_diff}")
    else:
        raise ValueError("mode should be in ['v1', 'v2']")