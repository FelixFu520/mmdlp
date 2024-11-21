import cv2
import os
import cv2
import glob
import sys
import numpy as np
from tqdm import tqdm
import argparse

class Preprocess:
    def __init__(self, rgb=False) -> None:
        self.auto = False  
        self.scaleFill = False  
        self.scaleup = True  
        self.stride = 32  
        self.center = True  
        self.rgb = rgb

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        shape = img.shape[:2]  
        new_shape = (640, 640) 

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        ratio = r, r 
        new_unpad = int(round(shape[1] * r)), int(
            round(shape[0] * r)
        )  
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  

        if self.auto:  
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill: 
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            ) 

        if self.center: 
            dw /= 2  
            dh /= 2  

        if shape[::-1] != new_unpad:  
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        img = img[np.newaxis, ...]

        im = img[..., ::-1].transpose(
            (0, 3, 1, 2)
        ) 
        im = im.astype(np.float32)  
        im = np.ascontiguousarray(im)
        
        if self.rgb:
            return im 
        
        im /= 255  
        return im 


def test_yolox_cali_data_rgb(val2017, calibration_data):
   
    os.makedirs(calibration_data, exist_ok=True)
    preobj = Preprocess(rgb=True)
    for i, img_path in enumerate(glob.glob(os.path.join(val2017, "*.jpg"))):
        # print(i, img_path)
        input_data = preobj.preprocess_image(img_path)
        img_name = os.path.basename(img_path)[:-4]
        dst_path = os.path.join(calibration_data, f"{img_name}.bin")
        print(i, dst_path)
        input_data.tofile(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="./coco_val2017_80_images")
    parser.add_argument("--calibration_data", type=str, default="./calibration_data_rgb")
    args = parser.parse_args()

    # TODO: Modify path 
    val2017 = args.images
    calibration_data = args.calibration_data
    test_yolox_cali_data_rgb(val2017, calibration_data)
     

