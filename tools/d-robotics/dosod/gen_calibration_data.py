import os
import cv2
import copy
import numpy as np
import argparse
from numpy import ndarray
from tqdm import tqdm
import os.path as osp
from matplotlib.figure import Figure
import onnxruntime as ort
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

from preprocess import preprocess_custom_v1, preprocess_custom_v2


def gen_calibration_data_v1_rgb_featuremap(data_dir: str, save_dir: str, height: int = 640, width: int = 2048):
    all_images = []
    for classes_name in os.listdir(data_dir):
        images = [os.path.join(data_dir, classes_name, p) for p in os.listdir(os.path.join(data_dir,classes_name)) if p.endswith("jpg") or p.endswith("JPG")]
        all_images.extend(images)
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_custom_v1(
            image_path, 
            height=height, 
            width=width,
        )
        image = image[np.newaxis, ...]
        assert image.shape == (1, 3, height, width)
        image.tofile(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".npy"))

def gen_calibration_data_v2_yuv444_featuremap(data_dir: str, save_dir: str, height: int = 640, width: int = 2048):
    all_images = []
    for classes_name in os.listdir(data_dir):
        images = [os.path.join(data_dir, classes_name, p) for p in os.listdir(os.path.join(data_dir,classes_name)) if p.endswith("jpg") or p.endswith("JPG")]
        all_images.extend(images)
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_custom_v2(
            image_path, 
            height=height, 
            width=width,
        )
        image = image[np.newaxis, ...]
        assert image.shape == (1, 3, height, width)
        image.tofile(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Calibration Data")
    parser.add_argument("--data_dir", type=str,  default= "./",  help="The directory of calibration images")
    parser.add_argument("--height", type=int, default=672,  help="height")
    parser.add_argument("--width", type=int, default=896, help="width")
    parser.add_argument("--preprocess", type=str, default="v1", help="v1: preprocess_custom_v1, v2: preprocess_custom_v2")
    parser.add_argument("--train", type=str, default="rgb", help="rgb,bgr,yuv444")
    parser.add_argument("--rt", type=str, default="featuremap", help="nv12,featuremap")
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.data_dir)
    save_dir = os.path.join(save_dir, f"{os.path.basename(args.data_dir)}_{args.preprocess}_{args.train}_{args.rt}_{args.height}×{args.width}")

    if args.preprocess == "v1" and args.train == "featuremap" and args.rt == "featuremap":
        gen_calibration_data_v1_rgb_featuremap(args.data_dir, save_dir=save_dir, height=args.height, width=args.width)
    elif args.preprocess == "v2" and args.train == "featuremap" and args.rt == "featuremap":
        gen_calibration_data_v2_yuv444_featuremap(args.data_dir, save_dir=save_dir, height=args.height, width=args.width)