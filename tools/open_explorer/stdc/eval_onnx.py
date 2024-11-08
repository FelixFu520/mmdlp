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
from postprocess import draw_sem_seg, classes, palette

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

SAVE_TEMP = False


def collect_val_data(datasets_dir: str):
    images_path = []

    for city_path in [os.path.join(datasets_dir,c) for c in os.listdir(datasets_dir)]:
        images_path = [os.path.join(city_path, tt) for tt in os.listdir(city_path)]
        for image_path in images_path:
            if image_path.endswith(".png"):
                yield image_path

def collect_val_data_list(datasets_dir: str):
    images_path = []

    for city_path in [os.path.join(datasets_dir,c) for c in os.listdir(datasets_dir)]:
        images_path = [os.path.join(city_path, tt) for tt in os.listdir(city_path)]
        for image_path in images_path:
            if image_path.endswith(".png"):
                images_path.append(image_path)
    return images_path


def eval_float_onnx(onnx_float_path, image_path, save_dir, height=1024, width=2048):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_float_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    image = preprocess_image(image_path, mean_std=True, MEAN=MEAN, STD=STD, bgr_to_rgb=True, transpose=True, height=height, width=width)

    # infer
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)

    # save
    output = outputs[0]
    np.save(osp.join(save_dir, osp.basename(image_path)[:-4] + ".npy"), output)

    # debug 存储一些结果
    if SAVE_TEMP:
        result_dir = osp.join(save_dir, "result_show")
        softmax_output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        label = np.argmax(softmax_output, axis=1).astype(np.uint8).squeeze(axis=0)
        label = cv2.resize(label, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        scores = np.max(softmax_output, axis=1).squeeze(axis=0)
        scores = cv2.resize(scores, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            # 保存labelids
            cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_labelids.png")), label)
            # 保存scores
            cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_scores.png")), (scores*255).astype(np.uint8))
            # 保存color_wrapper
            draw_sem_seg(label, classes, palette, image=cv2.resize(cv2.imread(image_path), (width, height)), alpha=0.5, 
                            out_file=os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_color.png")))


def eval_quant_onnx(onnx_quant_path, image_path, save_dir, height=1024, width=2048):
    os.makedirs(save_dir, exist_ok=True)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_quant_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_image(image_path, mean_std=False, to_float=True, 
                             bgr_to_rgb=True, transpose=True, new_axis=False, 
                             height=height, width=width)
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

    # save
    output = outputs[0]
    np.save(osp.join(save_dir, osp.basename(image_path)[:-4] + ".npy"), output)

    # debug 存储一些结果
    if SAVE_TEMP:
        result_dir = osp.join(save_dir, "result_show")
        softmax_output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        label = np.argmax(softmax_output, axis=1).astype(np.uint8).squeeze(axis=0)
        label = cv2.resize(label, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        scores = np.max(softmax_output, axis=1).squeeze(axis=0)
        scores = cv2.resize(scores, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            # 保存labelids
            cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_labelids.png")), label)
            # 保存scores
            cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_scores.png")), (scores*255).astype(np.uint8))
            # 保存color_wrapper
            draw_sem_seg(label, classes, palette, image=cv2.resize(cv2.imread(image_path), (width, height)), alpha=0.5, 
                            out_file=os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_color.png")))


def eval_all_onnx(onnx_float_path, onnx_quant_path, image_path, save_dir_float, save_dir_quant):
    eval_float_onnx(onnx_float_path, image_path, save_dir_float)
    eval_quant_onnx(onnx_quant_path, image_path, save_dir_quant)

if __name__ == "__main__":
    # 所有数据迭代器
    data_dir = "/horizon-bucket/aidi_public_data/cityscapes/origin/leftImg8bit/val"
    # all_val_images_path = collect_val_data_list(data_dir)
    all_val_images_path = collect_val_data(data_dir)

    
    # 使用float模型进行推理
    onnx_float_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/STDC1_pre.onnx"
    onnx_quant_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/output-stdc1_pre_1024×2048_jj/stdc1_pre_1024×2048_jj_quantized_model.onnx"
    save_dir_float = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_float_output2"
    save_dir_quant = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_quant_output3"
    
    print("start evaluating...")
    
    # 使用with语句确保进程池正确关闭
    with ProcessPoolExecutor(max_workers=32) as executor:
        for image_path in tqdm(all_val_images_path, desc="evaluating"):
            executor.submit(eval_all_onnx, onnx_float_path, onnx_quant_path, image_path, save_dir_float, save_dir_quant)
