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

from preprocess import preprocess_custom


def gen_calibration_data_featuremap(data_dir: str, save_dir: str, height: int = 640, width: int = 2048):
    all_images = []
    for classes_name in os.listdir(data_dir):
        images = [os.path.join(data_dir, classes_name, p) for p in os.listdir(os.path.join(data_dir,classes_name)) if p.endswith("jpg") or p.endswith("JPG")]
        all_images.extend(images)
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_custom(
            image_path, 
            height=height, 
            width=width,
        )
        image = image[np.newaxis, ...]
        assert image.shape == (1, 3, height, width)
        image.tofile(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".npy"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Calibration Data")
    parser.add_argument("--data_dir", type=str, 
                        default= "/home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_80", 
                        help="The directory of calibration images")
    parser.add_argument("--save_dir", type=str, 
                        default="/home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_80_featuremap_672×896",
                        help="The directory to save calibration data")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")
    args = parser.parse_args()

    gen_calibration_data_featuremap(args.data_dir, args.save_dir, height=args.height, width=args.width)