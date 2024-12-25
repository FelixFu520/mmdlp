import os
import os.path as osp
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

from prepcocess import preprocess_image

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

SAVE_TEMP = True
SCORES = 0.01
IOU_THRESHOLD = 0.5



def collect_val_data(datasets_dir: str):
    images_path = []

    for city_path in [os.path.join(datasets_dir,c) for c in os.listdir(datasets_dir)]:
        images_path = [os.path.join(city_path, tt) for tt in os.listdir(city_path)]
        for image_path in images_path:
            if image_path.endswith(".png"):
                yield image_path

def collect_val_data_list(datasets_dir: str):
    images_paths = []

    for image_path in [os.path.join(datasets_dir, t) for t in os.listdir(datasets_dir)]:
        images_paths.append(image_path)
    return images_paths


def eval_float_onnx(onnx_float_path, image_path, save_dir, height=1024, width=2048, show_dir = "eval_result_show"):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_float_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
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


    # infer
    feed_dict = {
        input_names[0]: image / 255,
    }
    outputs = sess.run(output_names, feed_dict)
    scores, bboxes = outputs

    # save
    output_path = os.path.join(save_dir, os.path.basename(image_path)[: -4])
    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/cls_scores.npy", scores)
    np.save(f"{output_path}/bbox_preds.npy", bboxes)

    # debug 存储一些结果
    if SAVE_TEMP:
        # NMS
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, SCORES, IOU_THRESHOLD)

        # 画图
        image = image.transpose(0, 2, 3, 1)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            # 显示score
            cv2.putText(image, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        result_dir = os.path.join(os.path.dirname(save_dir), show_dir)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_float.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)

def eval_calib_onnx(onnx_float_path, image_path, save_dir, height=1024, width=2048, show_dir = "eval_result_show"):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_float_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
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


    # infer
    feed_dict = {
        input_names[0]: image / 255,
    }
    outputs = sess.run(output_names, feed_dict)
    scores, bboxes = outputs

    # save
    output_path = os.path.join(save_dir, os.path.basename(image_path)[: -4])
    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/cls_scores.npy", scores)
    np.save(f"{output_path}/bbox_preds.npy", bboxes)

    # debug 存储一些结果
    if SAVE_TEMP:
        # NMS
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, SCORES, IOU_THRESHOLD)

        # 画图
        image = image.transpose(0, 2, 3, 1)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            # 显示score
            cv2.putText(image, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        result_dir = os.path.join(os.path.dirname(save_dir), show_dir)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_calib.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)

def eval_quant_onnx(onnx_float_path, image_path, save_dir, height=1024, width=2048, show_dir = "eval_result_show"):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_float_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
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


    # infer
    feed_dict = {
        input_names[0]: image / 255,
    }
    outputs = sess.run(output_names, feed_dict)
    scores, bboxes = outputs

    # save
    output_path = os.path.join(save_dir, os.path.basename(image_path)[: -4])
    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/cls_scores.npy", scores)
    np.save(f"{output_path}/bbox_preds.npy", bboxes)

    # debug 存储一些结果
    if SAVE_TEMP:
        # NMS
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, SCORES, IOU_THRESHOLD)

        # 画图
        image = image.transpose(0, 2, 3, 1)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            # 显示score
            cv2.putText(image, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        result_dir = os.path.join(os.path.dirname(save_dir), show_dir)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_quant.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)


def eval_all_onnx(
        onnx_float_path, onnx_quant_path, onnx_calib_path,
        image_path, 
        save_dir_float, save_dir_quant, save_dir_calib,
        height=1024, width=2048, 
        show_dir = "eval_result_show"
        ):
    eval_float_onnx(onnx_float_path, image_path, save_dir_float, height=height, width=width, show_dir=show_dir)
    eval_calib_onnx(onnx_calib_path, image_path, save_dir_calib, height=height, width=width, show_dir=show_dir)
    eval_quant_onnx(onnx_quant_path, image_path, save_dir_quant, height=height, width=width, show_dir=show_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate onnx model")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103", 
                        help="The directory of evaluation images")
    parser.add_argument("--onnx_float_path", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx",
                        help="The path of float onnx model")
    parser.add_argument("--onnx_calib_path", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/output/DOSOD_L_without_nms_v0.1_calib.onnx",
                        help="The path of calibration onnx model")
    parser.add_argument("--onnx_quant_path", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/output/DOSOD_L_without_nms_v0.1_quantized_model.onnx",
                        help="The path of quantized onnx model")
    parser.add_argument("--save_dir_float", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float",
                        help="The directory to save float model result")
    parser.add_argument("--save_dir_calib", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_calib",
                        help="The directory to save calibration model result")
    parser.add_argument("--save_dir_quant", type=str,
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant",
                        help="The directory to save quantized model result")
    parser.add_argument("--show_dir", type=str,
                        default="eval_result_show",
                        help="The directory to save show result")
    parser.add_argument("--height", type=int,
                        default=640,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=640,
                        help="width")
    args = parser.parse_args()

    # 所有数据迭代器
    data_dir = args.data_dir
    all_val_images_path = collect_val_data_list(data_dir)

    
    # 使用float模型进行推理
    onnx_float_path = args.onnx_float_path
    onnx_calib_path = args.onnx_calib_path
    onnx_quant_path = args.onnx_quant_path
    save_dir_float = args.save_dir_float
    save_dir_calib = args.save_dir_calib
    save_dir_quant = args.save_dir_quant
    
    print("start evaluating...")
    
    # for image_path in tqdm(all_val_images_path, desc="evaluating"):
    #     eval_all_onnx(onnx_float_path, onnx_quant_path, onnx_calib_path,
    #                   image_path, 
    #                   save_dir_float, save_dir_quant, save_dir_calib,
    #                   height=args.height, width=args.width, 
    #                   show_dir=args.show_dir)
    
    # 使用with语句确保进程池正确关闭
    with ProcessPoolExecutor(max_workers=32) as executor:
        for image_path in tqdm(all_val_images_path, desc="evaluating"):
            executor.submit(eval_all_onnx, 
                            onnx_float_path, onnx_quant_path, onnx_calib_path,
                            image_path, 
                            save_dir_float, save_dir_quant, save_dir_calib,
                            height=args.height, width=args.width, show_dir=args.show_dir)
