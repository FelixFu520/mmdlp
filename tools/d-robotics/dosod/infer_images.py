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

from preprocess import preprocess_custom_v1, preprocess_custom_v2


def collect_val_data_list(datasets_dir: str):
    images_paths = []

    for image_path in [os.path.join(datasets_dir, t) for t in os.listdir(datasets_dir)]:
        images_paths.append(image_path)
    return images_paths


def post_process(image_show, outputs, image_path:str, result_dir: str, onnx_type:str = "float", score_threshold=0.01, iou_threshold=0.5):
    """
    image_show: [1, 3, H, W], BGR
    outputs: [1, N, num_classes], [1, N, 4]
    """
    assert image_show.shape[0] == 1, "only support batch size 1"
    assert image_show.shape[1] == 3, "only support 3 channels"
    assert image_path.endswith(".png") or image_path.endswith(".jpg"), "image_path must be a .png or .jpg file"
    assert result_dir, "result_dir must be a valid directory"

    # NMS
    scores, bboxes = outputs
    bboxes = bboxes.squeeze(0)
    scores = scores.squeeze(0)
    argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
    argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
    indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, score_threshold, iou_threshold)

    # 画图
    image_show = image_show.transpose(0, 2, 3, 1)
    image_show = cv2.cvtColor(image_show[0], cv2.COLOR_RGB2BGR)
    for idx in indexs:
        cv2.rectangle(image_show, (int(bboxes[idx][0]), int(bboxes[idx][1])), (int(bboxes[idx][2]), int(bboxes[idx][3])), (0, 255, 0), 2)
        cv2.putText(image_show, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4] + f"_result_{onnx_type}.png")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, image_show)


def infer_onnx(onnx_model_path:str, image_path:str, preprocess_fun: callable, postprocess_fn:callable):
    # preprocess image
    image = preprocess_fun(image_path)

    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # infer
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)

    # post process
    postprocess_fn(image, outputs, image_path)


def infer_image(args):
    if args.onnx_float_path:
        def preprocess_fix(image_path, height=args.height, width=args.width) -> np.ndarray:
            if args.preprocess == "v1":
                image = preprocess_custom_v1(image_path, height, width)
                image = np.expand_dims(image, axis=0)   # [1, 3, H, W], RGB, 0-1
            # elif args.preprocess == "v2":
            #     preprocess_custom_v2(image_path, height, width)
            #     image = np.expand_dims(image, axis=0)
            else:
                raise ValueError("args error")
            return image  
        def postprocess_fix(image, outputs, image_path, work_dir=args.work_dir, onnx_type="float", score_threshold=args.score_threshold, iou_threshold=args.iou_threshold) -> None:
            save_dir_path = os.path.join(work_dir, "demo_result")
            if args.preprocess == "v1":
                image_show = (image * 255).astype(np.uint8) # [1, 3, H, W], RGB, 0-255
                post_process(image_show, outputs, image_path, save_dir_path, onnx_type, score_threshold, iou_threshold)
            # elif args.preprocess == "v2":
            #     post_process(image_show, outputs, image_path, save_dir_path, onnx_type, score_threshold, iou_threshold)
            else:
                raise ValueError("args error")
        infer_onnx(args.onnx_float_path, args.image_path, preprocess_fix, postprocess_fix)
    
    if args.onnx_origin_path:
        def preprocess_fix(image_path, height=args.height, width=args.width) -> np.ndarray:
            if args.preprocess == "v1":
                image = preprocess_custom_v1(image_path, height, width)
                image = image * 255
                fun_t = RGB2YUV444Transformer(data_format="CHW")
                image = fun_t(image)
                image = np.expand_dims(image, axis=0)
            # elif args.preprocess == "v2":
            #     preprocess_custom_v2(image_path, height, width)
            #     image = np.expand_dims(image, axis=0)
            else:
                raise ValueError("args error")
            return image
        def postprocess_fix(image, outputs, image_path, work_dir=args.work_dir, onnx_type="origin", score_threshold=args.score_threshold, iou_threshold=args.iou_threshold) -> None:
            save_dir_path = os.path.join(work_dir, "demo_result")
            if args.preprocess == "v1":
                image_show = image.astype(np.uint8)
                post_process(image_show, outputs, image_path, save_dir_path, onnx_type, score_threshold, iou_threshold)
            elif args.preprocess == "v2":
                post_process(image_show, outputs, image_path, save_dir_path, onnx_type, score_threshold, iou_threshold)
            else:
                raise ValueError("args error")
        infer_onnx(args.onnx_origin_path, args.image_path, preprocess_fix, postprocess_fix)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer images")
    parser.add_argument("--images_path", type=str, default="",  help="The directory of evaluation images")
    parser.add_argument("--image_path", type=str, default="", help="The path of evaluation image")

    parser.add_argument("--onnx_float_path", type=str, default="", help="The path of float onnx model")
    parser.add_argument("--onnx_origin_path", type=str, default="", help="The path of origin onnx model")
    parser.add_argument("--onnx_optim_path", type=str, default="", help="The path of optimized onnx model")
    parser.add_argument("--onnx_calib_path", type=str, default="", help="The path of calibration onnx model")
    parser.add_argument("--onnx_quant_path", type=str, default="", help="The path of quantized onnx model")

    parser.add_argument("--save_dir_float", type=str, default="", help="The directory to save float model result")
    parser.add_argument("--save_dir_origin", type=str, default="", help="The directory to save origin model result")
    parser.add_argument("--save_dir_optim", type=str, default="", help="The directory to save optim model result")
    parser.add_argument("--save_dir_calib", type=str, default="", help="The directory to save calibration model result")
    parser.add_argument("--save_dir_quant", type=str, default="", help="The directory to save quantized model result")


    parser.add_argument("--onnx_calib_featuremap_path", type=str, default="")
    parser.add_argument("--onnx_quant_featuremap_path", type=str, default="")

    parser.add_argument("--save_dir_calib_featuremap", type=str, default="")
    parser.add_argument("--save_dir_quant_featuremap", type=str, default="")

    parser.add_argument("--work_dir", type=str, default="", help="The directory to save show result")
    parser.add_argument("--height", type=int, default=672, help="height")
    parser.add_argument("--width", type=int, default=896, help="width")
    parser.add_argument("--show", type=bool, default=True, help="show result")
    parser.add_argument("--score_threshold", type=float, default="0.01", help="score")
    parser.add_argument("--iou_threshold", type=float, default="0.5", help="iou_threshold")
    parser.add_argument("--preprocess", type=str, default="v1", help="v1: preprocess_custom_v1, v2: preprocess_custom_v2")
    parser.add_argument("--multi_process", type=bool, default=False, help="multi process")

    args = parser.parse_args()


    if args.image_path:
        all_images_path = [args.image_path]
    elif args.images_path:
        all_images_path = collect_val_data_list(args.images_path)
    else:
        raise ValueError("args error")
    
    if args.multi_process:
        with ProcessPoolExecutor(max_workers=32) as executor:
            for image_path in tqdm(all_images_path, desc="evaluating"):
                executor.submit(infer_image, image_path, args)
    else:
        for image_path in tqdm(all_images_path, desc="evaluating"):
            infer_image(args)
