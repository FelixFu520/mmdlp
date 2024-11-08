import os
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime

from prepcocess import preprocess_image
from postprocess import postprocess

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]


def collect_val_data(datasets_dir: str) -> list:
    images_path = []

    for city_path in [os.path.join(datasets_dir,c) for c in os.listdir(datasets_dir)]:
        images_path = [os.path.join(city_path, tt) for tt in os.listdir(city_path)]
        for image_path in images_path:
            if image_path.endswith(".png"):
                images_path.append(image_path)

def eval_float_onnx(onnx_path, img_path):
    pass


if __name__ == "__main__":
    data_dir = "/horizon-bucket/aidi_public_data/cityscapes/origin/leftImg8bit/val"
    all_val_images_path = collect_val_data(data_dir)
    pass