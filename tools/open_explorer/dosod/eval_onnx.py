import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

from prepcocess import preprocess_image

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

SAVE_TEMP = True


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


def eval_float_onnx(onnx_float_path, image_path, save_dir, height=1024, width=2048):
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
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.6, 0.5)

        # 画图
        image = image.transpose(0, 2, 3, 1)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
        result_dir = os.path.join(os.path.dirname(save_dir), "eval_result_show")
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_float.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)


def eval_quant_onnx(onnx_quant_path, image_path, save_dir, height=1024, width=2048):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_quant_path)
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
    scores, bboxes = outputs

    # save
    output_path = os.path.join(save_dir, os.path.basename(image_path)[: -4])
    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/cls_scores.npy", scores)
    np.save(f"{output_path}/bbox_preds.npy", bboxes)

    # debug 存储一些结果
    if SAVE_TEMP:
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.6, 0.5)

        # 画图
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
        result_dir = os.path.join(os.path.dirname(save_dir), "eval_result_show")
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_quant.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)


def eval_all_onnx(onnx_float_path, onnx_quant_path, image_path, save_dir_float, save_dir_quant, height=1024, width=2048):
    eval_float_onnx(onnx_float_path, image_path, save_dir_float, height=height, width=width)
    eval_quant_onnx(onnx_quant_path, image_path, save_dir_quant, height=height, width=width)

if __name__ == "__main__":
    # 所有数据迭代器
    data_dir = "/home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103"
    all_val_images_path = collect_val_data_list(data_dir)

    
    # 使用float模型进行推理
    onnx_float_path = "/home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx"
    onnx_quant_path = "/home/users/fa.fu/work/work_dirs/dosod/20241103/output/DOSOD_L_without_nms_v0.1_quantized_model.onnx"
    save_dir_float = "/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float"
    save_dir_quant = "/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant"
    
    print("start evaluating...")
    
    # for image_path in tqdm(all_val_images_path, desc="evaluating"):
    #     eval_all_onnx(onnx_float_path, onnx_quant_path, image_path, save_dir_float, save_dir_quant, height=640, width=640)
    
    # 使用with语句确保进程池正确关闭
    with ProcessPoolExecutor(max_workers=32) as executor:
        for image_path in tqdm(all_val_images_path, desc="evaluating"):
            executor.submit(eval_all_onnx, onnx_float_path, onnx_quant_path, image_path, save_dir_float, save_dir_quant, height=640, width=640)
