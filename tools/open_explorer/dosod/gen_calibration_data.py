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

from prepcocess import preprocess_image

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]


def gen_calibration_data(data_dir: str, save_dir: str, height: int = 640, width: int = 2048):
    all_images = []
    for classes_name in os.listdir(data_dir):
        images = [os.path.join(data_dir, classes_name, p) for p in os.listdir(os.path.join(data_dir,classes_name)) if p.endswith("jpg") or p.endswith("JPG")]
        all_images.extend(images)
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_image(
            image_path, 
            height=height, 
            width=width,
            bgr_to_rgb=True,
            to_float=True,
            mean_std=False, 
            MEAN=MEAN, 
            STD=STD,  
            transpose=True, 
            new_axis=True
        )
        image = image
        assert image.shape == (1, 3, height, width)
        image.tofile(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".npy"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Calibration Data")
    parser.add_argument("--data_dir", type=str, 
                        default= "/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1", 
                        help="The directory of calibration images")
    parser.add_argument("--save_dir", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb",
                        help="The directory to save calibration data")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")
    args = parser.parse_args()

    gen_calibration_data(args.data_dir, args.save_dir, height=args.height, width=args.width)