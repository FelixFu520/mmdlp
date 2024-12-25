import os
import shutil
import numpy as np
import argparse
import random

def collect_calibrate_data(datasets_dir, calibrate_data_dir, sample_number):
    # os.makedirs(calibrate_data_dir, exist_ok=True)

    samples = os.listdir(datasets_dir)
    samples_len = len(samples)
    copy_num = 0
    print('dataset length ', samples_len, ", select ", sample_number)
    for sample_path in [os.path.join(datasets_dir, s) for s in samples]:
        if copy_num < sample_number:
            random_num = random.randint(0, samples_len)
            print("select ", random_num , ", ", os.path.basename(sample_path),  ", as calibration dataset")
            shutil.copy(sample_path, calibrate_data_dir)
            copy_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect calibration data for calibration.")
    parser.add_argument("--datasets_dir",
                        default="/horizon-bucket/AIoT-data-bucket/AIOT_algorithm_data/train_stain_dataset/real_resize_jpg_data_20241103",
                        type=str, required=False, help="Path to the datasets dir.")
    parser.add_argument("--calibrate_data_dir",
                        default="/home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1",
                        type=str, required=False, help="Path to the calibrate data dir.")
    parser.add_argument("--number",
                        default=140,
                        type=int, required=False, help="sample number.")
    
    args = parser.parse_args()
                        
    collect_calibrate_data(args.datasets_dir, args.calibrate_data_dir, args.number)
    print("Done!")

    