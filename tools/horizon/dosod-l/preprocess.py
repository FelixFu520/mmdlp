import argparse
import cv2
import os
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union
from mmcv.transforms import LoadAnnotations, LoadImageFromFile
from mmyolo.datasets.transforms import YOLOv5KeepRatioResize, LetterResize


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

def preprocess_custom(image_path: str, height: int = 672, width: int = 896, 
                      allow_scale_up: bool = False,
                      pad_val = 114):
    """
    主要使用opencv对图像进行前处理
    1. read
    2. gray2bgr
    3. resize
    4. bgr2rgb
    5. to_float
    6. mean_std
    7. transpose
    8. add new_axis
    9. ascontiguousarray
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



def preprocess_mmcv(image_path: str, height: int = 672, width: int = 896):
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
    input_data = image['img']
    input_data = input_data.astype(np.float32)
    input_data = input_data[..., ::-1]
    input_data /= 255
    input_data = input_data.transpose(2, 0, 1)

    assert input_data.shape == (3, height, width)

    return input_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare input")
    parser.add_argument("--images_dir", type=str, default="/home/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103")
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=896)

    args = parser.parse_args()


    images_path = [os.path.join(args.images_dir, img) for img in os.listdir(args.images_dir) if img.endswith('.jpg')]
    for img_p in tqdm(images_path, desc="compare input"):
        image = preprocess_custom(img_p, height=args.height, width=args.width)
        image_mmcv = preprocess_mmcv(img_p, height=args.height, width=args.width)

        max_diff = np.sum(np.abs(image - image_mmcv))
        if max_diff > 0:
            print(f"{os.path.basename(img_p)} max_diff: {max_diff}")