import os
import cv2
import copy
import numpy as np
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


def gen_calibration_data(data_dir: str, save_dir: str, height: int = 1024, width: int = 2048):
    all_images = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith("png")]
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_image(image_path, height=height, width=width, bgr_to_rgb=True, mean_std=True, MEAN=MEAN, STD=STD, transpose=True, new_axis=True)
        assert image.shape == (1, 3, height, width)
        image.tofile(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".bin"))

if __name__ == "__main__":
    data_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/calibrate_data_dir"
    save_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/calibration_data_rgb_1024×2048"
    gen_calibration_data(data_dir, save_dir)