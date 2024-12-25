import numpy as np
import argparse
from mmcv.transforms import LoadAnnotations, LoadImageFromFile
from mmyolo.datasets.transforms import YOLOv5KeepRatioResize, LetterResize
from prepcocess import preprocess_image
MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare input")
    parser.add_argument("--img_path", type=str, default="/home/fa.fu/work/work_dirs/dosod_debug/demo_images/0892.jpg")
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=896)
    args = parser.parse_args()

    img_path = args.img_path
    height = args.height
    width = args.width


    # 准备pth输入的数据, 从训练的配置文件中抄过来的
    img_path_dict = {"img_path": img_path}
    imgs = LoadImageFromFile().transform(img_path_dict)
    img_scale = (width, height)
    imgs = YOLOv5KeepRatioResize(scale=img_scale).transform(imgs)
    imgs = LetterResize(scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)).transform(imgs)
    input_data = imgs["img"]
    input_data = input_data.astype(np.float32) 
    input_data = input_data[..., ::-1]
    input_data /= 255
    input_data = input_data.transpose(2, 0, 1)


    # 准备onnx输出的数据
    image = preprocess_image(
        img_path, 
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
    image = image/255
    
    # 对比
    max_diff = np.abs(input_data - image).max()
    print("max_diff:", max_diff)
