import os
from tqdm import tqdm
import shutil
import numpy as np


def collect_calibrate_data(datasets_dir, calibrate_data_dir, class_number):
    os.makedirs(calibrate_data_dir, exist_ok=True)

    cities = os.listdir(datasets_dir)
    for city_path in [os.path.join(datasets_dir,c) for c in cities]:
        images_path = [os.path.join(city_path, tt) for tt in os.listdir(city_path)]

        image_num = 0
        for image_path in images_path:
            if image_num > class_number:
                break
            shutil.copy(image_path, calibrate_data_dir)
            image_num += 1


if __name__ == "__main__":
    left_train_data = f"/home/users/fa.fu/work/data/cityscapes/leftImg8bit/train"
    right_train_data = f"/home/users/fa.fu/work/data/cityscapes/rightImg8bit/train"
    calibrate_data_dir = f"/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/calibrate_data_dir"
    number = 10
    collect_calibrate_data(left_train_data, calibrate_data_dir, 10)
    collect_calibrate_data(left_train_data, calibrate_data_dir, 10)
    print(f"total size: {len(os.listdir(calibrate_data_dir))}")
    print("Done!")

    