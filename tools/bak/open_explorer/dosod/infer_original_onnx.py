import os
import cv2
import numpy as np
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime
import argparse

from prepcocess import preprocess_image

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

def infer_origin_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height=672, width=896):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
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
        input_names[0]: (image/255),
    }
    outputs = sess.run(output_names, feed_dict)
    
    # 后处理
    if result_dir is not None:
        # NMS
        scores, bboxes = outputs
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
        argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.4, 0.5)

        # 画图
        image = image.transpose(0, 2, 3, 1)
        image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)
    
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer_origin_onnx")
    parser.add_argument("--onnx_path", type=str, 
                        default= "/home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l-epoch_40_kxj_rep-without-nms_20241023_896x672.onnx", 
                        help="onnx path")
    parser.add_argument("--image_path", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/demo_images/030125.jpg",
                        help="image path")
    parser.add_argument("--result_dir", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/dosod/result",
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

    # 使用原始onnx推理查看下onnx是否正确
    infer_origin_onnx(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
        height=args.height,
        width=args.width,
    )