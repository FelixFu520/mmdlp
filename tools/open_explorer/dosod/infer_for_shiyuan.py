# 推理一批图， 转成固定格式给世源, 对应的数据存在work_dirs/dosod/demo_images/infer_for_shiyuan/

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
import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

from prepcocess import preprocess_image

classes = (
            'liquid stain',
            'congee stain',
            'milk stain',
            'skein',
            'solid stain',
)

def infer_quant_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    
    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_image(
        image_path, 
        height=height, 
        width=width,
        bgr_to_rgb=True,
        to_float=True,
        mean_std=False, 
        transpose=True, 
        new_axis=False
    )
    fun_t = RGB2YUV444Transformer(data_format="CHW")
    input_data = fun_t.run_transform(image)
    input_data = input_data[np.newaxis, ...]
    input_data -= 128
    input_data = input_data.astype(np.int8)
    input_data = input_data.transpose(0, 2, 3, 1)
    # infer
    feed_dict = {
        input_names[0]: input_data,
    }
    outputs = sess.run(output_names, feed_dict)

    if result_dir is not None:
        # NMS
        scores, bboxes = outputs
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.05, 0.5)

        # 画图
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 存储xml
        # 创建根元素annotation
        annotation = ET.Element("annotation")
        # 创建folder元素并添加文本内容
        folder = ET.SubElement(annotation, "folder")
        folder.text = "VOCImages"
        # 创建filename元素并添加文本内容（这里假设你可以通过变量传入具体文件名）
        filename = ET.SubElement(annotation, "filename")
        filename.text = f"img_baihe_wuzi/{os.path.basename(image_path)}"
        # 创建size元素及子元素width、height、depth，并添加对应文本内容（示例中的尺寸，可根据实际修改）
        image_raw = cv2.imread(image_path)
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_raw.shape[1])
        height = ET.SubElement(size, "height")
        height.text = str(image_raw.shape[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(image_raw.shape[2])

        for idx in indexs:
            # 创建object元素以及它包含的各个子元素，并添加相应内容（示例中的目标对象信息，可按需修改）
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = classes[argmax_idx[idx]]

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            score = ET.SubElement(obj, "score")
            score.text = str(argmax_scores[idx])

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(bboxes[idx][0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(bboxes[idx][1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(bboxes[idx][2])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(bboxes[idx][3])

            # 画图
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            cv2.putText(image, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        dst_image_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        cv2.imwrite(dst_image_path, image)

        dst_xml_path = os.path.join(os.path.dirname(result_dir), os.path.basename(image_path)[:-4]+".xml")
        tree = ET.ElementTree(annotation)
        tree.write(dst_xml_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer_quant_onnx")
    parser.add_argument("--onnx_path", type=str, 
                        default= "/home/users/fa.fu/work/work_dirs/dosod/20241113/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx", 
                        help="onnx path")
    parser.add_argument("--image_path", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/demo_images/infer_for_shiyuan/img_baihe_wuzi",
                        help="image path")
    parser.add_argument("--result_dir", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/demo_images/infer_for_shiyuan/result",
                        help="result dir")
    parser.add_argument("--height", type=int,
                        default=672,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=896,
                        help="width")

    args = parser.parse_args()

    onnx_model_path = args.onnx_path
    image_path = args.image_path
    result_dir = args.result_dir

    os.makedirs(result_dir, exist_ok=True)

    all_images = [os.path.join(image_path, t) for t in os.listdir(image_path) if t.endswith(".jpg")]

    # 使用原始onnx推理查看下onnx是否正确
    # for image_path in tqdm(all_images):
    #     infer_quant_onnx(
    #         onnx_model_path=onnx_model_path,
    #         image_path=image_path,
    #         result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
    #         height=args.height,
    #         width=args.width,
    #     )
    
    # 使用with语句确保进程池正确关闭
    with ProcessPoolExecutor(max_workers=64) as executor:
        for image_path in tqdm(all_images, desc="infer"):
            executor.submit(infer_quant_onnx, 
                            onnx_model_path=onnx_model_path, 
                            image_path=image_path, 
                            result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
                            height=args.height, 
                            width=args.width)