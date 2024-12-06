# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
import cv2
import numpy as np
from PIL import Image
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

class _ChannelSwapTransformer(BaseTransform):
    def __init__(self, order: tuple, channel_index: int = 0) -> None:
        self.order = order
        self.channel_index = channel_index

    def transform(self, results: dict) -> dict:
        image = results['img']
        assert self.channel_index < len(image.shape), \
            "channel index is larger than image.dims"
        assert image.shape[self.channel_index] == len(self.order), \
            "the length of swap order != the number of channel:{}!={}" \
            .format(len(self.order), image.shape[self.channel_index])
        if self.channel_index == 0:
            image = image[self.order, :, :]
        elif self.channel_index == 1:
            image = image[:, self.order, :]
        elif self.channel_index == 2:
            image = image[:, :, self.order]
        else:
            raise ValueError(f"channel index: {self.channel_index} error "
                             "in _ChannelSwapTransformer")
        results['img'] = image
        return results


@TRANSFORMS.register_module()
class BGR2RGBTransformer(BaseTransform):
    def __init__(self, data_format: str = "CHW") -> None:
        if data_format == "CHW":
            self.transformer = _ChannelSwapTransformer((2, 1, 0))
        elif data_format == "HWC":
            self.transformer = _ChannelSwapTransformer((2, 1, 0), 2)
        else:
            raise ValueError(f"unsupported data_format: '{data_format}' "
                             "in BGR2RGBTransformer")

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results


@TRANSFORMS.register_module()
class RGB2BGRTransformer(BaseTransform):
    def __init__(self, data_format="CHW"):
        if data_format == "CHW":
            self.transformer = _ChannelSwapTransformer((2, 1, 0))
        elif data_format == "HWC":
            self.transformer = _ChannelSwapTransformer((2, 1, 0), 2)
        else:
            raise ValueError(f"unsupported data_format: '{data_format}' "
                             "in RGB2BGRTransformer")

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results


def rgb2bt601_full_range(r, g, b, single_channel=False):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if not single_channel:
        u = -0.169 * r - 0.331 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v
    else:
        return y


def yuv444_to_rgb(y, u, v):
    r = y + 1.402 * (v - 128)
    g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = y + 1.772 * (u - 128)

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    return r, g, b


def rgb2bt601_video_range(r, g, b, single_channel=False):
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    if not single_channel:
        u = -0.148 * r - 0.291 * g + 0.439 * b + 128
        v = 0.439 * r - 0.368 * g - 0.071 * b + 128
        return y, u, v
    else:
        return y


class _ColorConvertTransformer(BaseTransform):
    def __init__(self, source_type: str, target_type: str, data_format: str = "CHW") -> None:
        self.source_type = source_type
        self.target_type = target_type.upper()
        self.data_format = data_format.upper()
        # get source format and range
        source_format_range = self.source_type.split('_')
        self.source_format = source_format_range[0].upper()
        self.source_range = source_format_range[1] \
            if len(source_format_range) == 2 else '255'
        # get target format and range
        if self.target_type in [
                'YUV_BT601_VIDEO_RANGE', 'YUV_BT601_FULL_RANGE'
        ]:
            self.target_format = self.target_type
            self.target_range = '128'
        else:
            target_format_range = self.target_type.split('_')
            self.target_format = target_format_range[0]
            self.target_range = target_format_range[1] \
                if len(target_format_range) == 2 else '255'
        # all the color convert operated on data range in [0, 255]
        self.source_offset = 128. if self.source_range == "128" else 0.
        self.target_offset = -128. if self.target_range == "128" else 0.


    def transform_func(self, image) -> np.ndarray:
        if self.source_format == self.target_format:
            image = image
        else:
            # split source input to r, g, b
            if self.source_format == 'RGB' and self.data_format == 'HWC':
                image = (image[:, :, 0], image[:, :, 1], image[:, :, 2])
            elif self.source_format == 'RGB' and self.data_format == 'CHW':
                image = (image[0, :, :], image[1, :, :], image[2, :, :])
            elif self.source_format == 'BGR' and self.data_format == 'HWC':
                image = (image[:, :, 2], image[:, :, 1], image[:, :, 0])
            elif self.source_format == 'BGR' and self.data_format == 'CHW':
                image = (image[2, :, :], image[1, :, :], image[0, :, :])
            elif self.source_format == 'YUV444' and self.data_format == 'CHW':
                image = (image[0, :, :], image[1, :, :], image[2, :, :])
                image = yuv444_to_rgb(*image)
            elif self.source_format == 'YUV444' and self.data_format == 'HWC':
                image = (image[:, :, 0], image[:, :, 1], image[:, :, 2])
                image = yuv444_to_rgb(*image)
            else:
                ValueError(
                    f"Unknown color convert source_format:{self.source_format}"
                    f" or data_format{self.data_format}, please check yaml")
            # convert r, g, b to yuv or gray
            if self.target_format == 'RGB':
                image = image
            elif self.target_format == 'BGR':
                image = (image[2], image[1], image[0])
            elif self.target_format == 'YUV444' or \
                    self.target_format == 'YUV_BT601_FULL_RANGE':
                image = rgb2bt601_full_range(*image)
            elif self.target_format == 'YUV_BT601_VIDEO_RANGE':
                image = rgb2bt601_video_range(*image)
            elif self.target_format == 'GRAY':
                image = rgb2bt601_full_range(*image, single_channel=True)
            else:
                ValueError("Unknown color convert target_format: "
                           f"{self.target_format}, please check yaml")
            # fuse convert result(b, g, r or y, u, v) to target output
            if self.data_format == 'HWC':
                if self.target_format == 'GRAY':
                    image = image[:, :, np.newaxis]
                else:
                    image = np.array(image).transpose((1, 2, 0))
            elif self.data_format == 'CHW':
                if self.target_format == 'GRAY':
                    image = image[np.newaxis, :, :]
                else:
                    image = np.array(image)
        return image


    def transform(self, results: dict) -> dict:
        image = results['img'] + self.source_offset
        converted_image = self.transform_func(image)
        converted_image = converted_image + self.target_offset
        results['img'] = converted_image.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class RGB2YUV444Transformer(BaseTransform):
    def __init__(self, data_format: str = 'CHW') -> None:
        assert data_format in ['CHW', 'HWC'], "Data_format must in 'CHW' or 'HWC' "
        self.transformer = _ColorConvertTransformer('RGB', 'YUV444',
                                                    data_format)

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results


@TRANSFORMS.register_module()
class BGR2YUV444Transformer(BaseTransform):
    def __init__(self, data_format: str = 'CHW') -> None:
        self.transformer = _ColorConvertTransformer('BGR', 'YUV444',
                                                    data_format)

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results

@TRANSFORMS.register_module()
class YUV4442RGBTransformer(BaseTransform):
    def __init__(self, data_format: str = 'CHW') -> None:
        self.transformer = _ColorConvertTransformer('YUV444', 'RGB',
                                                    data_format)

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results

@TRANSFORMS.register_module()
class YUV4442BGRTransformer(BaseTransform):
    def __init__(self, data_format: str = 'CHW') -> None:
        self.transformer = _ColorConvertTransformer('YUV444', 'BGR',
                                                    data_format)

    def transform(self, results: dict) -> dict:
        results = self.transformer.transform(results)
        return results

@TRANSFORMS.register_module()
class BGR2NV12Transformer(BaseTransform):
    @staticmethod
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

    def __init__(self, data_format: str = "CHW", cvt_mode: str = 'rgb_calc') -> None:
        self.cvt_mode = cvt_mode
        self.data_format = data_format

    def rgb2nv12_calc(self, image):
        if image.ndim == 3:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return yuv.astype(np.uint8)
        else:
            raise ValueError("image is not BGR format")

    def rgb2nv12_opencv(self, image):
        if image.ndim == 3:
            image = image.astype(np.uint8)
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(
                (height * width * 3 // 2, ))
            y = yuv420p[:height * width]
            uv_planar = yuv420p[height * width:].reshape(
                (2, height * width // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2, ))
            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        else:
            raise ValueError("image is not BGR format")

    def transform(self, results: dict) -> dict:
        image = results['img']
        if self.data_format == "CHW":
            image = np.transpose(image, (1, 2, 0))

        image_shape = image.shape[:-1]
        if image_shape[0] * image_shape[1] % 2 != 0:
            raise ValueError(
                f"Invalid odd shape: {image_shape[0]} x {image_shape[1]}, "
                "expect even number for height and width")

        if self.cvt_mode == 'opencv':
            image = self.rgb2nv12_opencv(image)
        else:
            image = self.rgb2nv12_calc(image)
        
        results['img'] = image
        return results

@TRANSFORMS.register_module()
class RGB2NV12Transformer(BaseTransform):
    @staticmethod
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

    def __init__(self, data_format: str = "CHW", cvt_mode: str = 'rgb_calc') -> None:
        self.cvt_mode = cvt_mode
        self.data_format = data_format

    def rgb2nv12_calc(self, image):
        if image.ndim == 3:
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return yuv.astype(np.uint8)
        else:
            raise ValueError("image is not BGR format")

    def rgb2nv12_opencv(self, image):
        if image.ndim == 3:
            image = image.astype(np.uint8)
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (height * width * 3 // 2, ))
            y = yuv420p[:height * width]
            uv_planar = yuv420p[height * width:].reshape(
                (2, height * width // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2, ))
            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        else:
            raise ValueError("image is not BGR format")

    def transform(self, results: dict) -> dict:
        image = results['img']
        if self.data_format == "CHW":
            image = np.transpose(image, (1, 2, 0))

        image_shape = image.shape[:-1]
        if image_shape[0] * image_shape[1] % 2 != 0:
            raise ValueError(
                f"Invalid odd shape: {image_shape[0]} x {image_shape[1]}, "
                "expect even number for height and width")

        if self.cvt_mode == 'opencv':
            image = self.rgb2nv12_opencv(image)
        else:
            image = self.rgb2nv12_calc(image)
        results['img'] = image
        return results

@TRANSFORMS.register_module()
class NV12ToYUV444Transformer(BaseTransform):
    def __init__(self, target_size: tuple, yuv444_output_layout: str = "HWC") -> None:
        self.height = target_size[0]
        self.width = target_size[1]
        self.yuv444_output_layout = yuv444_output_layout

    def transform(self, results: dict) -> dict:
        image = results['img']
        nv12_image = image.flatten()
        yuv444 = np.empty([self.height, self.width, 3], dtype=np.uint8)
        yuv444[:, :, 0] = nv12_image[:self.width * self.height].reshape(
            self.height, self.width)
        u = nv12_image[self.width * self.height::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 1] = Image.fromarray(u).resize((self.width, self.height),
                                                    resample=0)
        v = nv12_image[self.width * self.height + 1::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 2] = Image.fromarray(v).resize((self.width, self.height),
                                                    resample=0)
        image = yuv444.astype(np.uint8)
        if self.yuv444_output_layout == "CHW":
            image = np.transpose(image, (2, 0, 1))
        results['img'] = image
        return results