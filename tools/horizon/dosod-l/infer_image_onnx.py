import os
import cv2
import numpy as np
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import RGB2YUV444Transformer, BGR2NV12Transformer, NV12ToYUV444Transformer, RGB2NV12Transformer
import argparse

from preprocess import preprocess_custom

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

def infer_origin_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height=672, width=896):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
        )
    image = np.expand_dims(image, axis=0)
    image_show = (image * 255).astype(np.uint8)
    # infer
    feed_dict = {
        input_names[0]: image,
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
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.01, 0.5)

        # 画图
        image_show = image_show.transpose(0, 2, 3, 1)
        image_show = cv2.cvtColor(image_show[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image_show, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            cv2.putText(image_show, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image_show)
    

def infer_calib_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
    )
    image = image * 255
    image = np.expand_dims(image, axis=0)
    image_show = image.astype(np.uint8)
    fun_t = RGB2YUV444Transformer(data_format="CHW")
    input_data = fun_t.run_transform(image[0])
    input_data = input_data[np.newaxis, ...]
    # input_data -= 128
    # input_data = input_data.astype(np.int8)
    # input_data = input_data.transpose(0, 2, 3, 1)

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
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.01, 0.5)

        # 画图
        image_show = image_show.transpose(0, 2, 3, 1)
        image_show = cv2.cvtColor(image_show[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image_show, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            cv2.putText(image_show, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image_show)

def infer_quant_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
    )
    image = image * 255
    image = np.expand_dims(image, axis=0)
    image_show = image.astype(np.uint8)

    # fun_t = RGB2YUV444Transformer(data_format="CHW")  # 这个是无损的和板端有区别, 替换成下面完全模拟板端的
    fun_t1 = RGB2NV12Transformer(data_format="CHW")
    fun_t2 = NV12ToYUV444Transformer((height, width), yuv444_output_layout="CHW")
    input_data = fun_t1.run_transform(image[0])
    input_data = fun_t2.run_transform(input_data)

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
        indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.01, 0.5)

        # 画图
        image_show = image_show.transpose(0, 2, 3, 1)
        image_show = cv2.cvtColor(image_show[0], cv2.COLOR_RGB2BGR)
        for idx in indexs:
            cv2.rectangle(image_show, 
                        (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                        (int(bboxes[idx][2]), int(bboxes[idx][3])),
                        (0, 255, 0), 
                        2)
            cv2.putText(image_show, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result.png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer_onnx")
    parser.add_argument("--onnx_float_path", type=str, 
                        default= "", 
                        help="onnx path")
    parser.add_argument("--onnx_path", type=str, 
                        default= "", 
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
    parser.add_argument('--mode', type=str, default="origin")
    args = parser.parse_args()

    onnx_model_path = args.onnx_float_path
    if args.onnx_path:
        onnx_model_path = args.onnx_path    # onnx_float_path和onnx_path是同一个参数, 为了兼容之前的内容所以加了一个
    image_path = args.image_path
    result_dir = args.result_dir

    os.makedirs(result_dir, exist_ok=True)

    if args.mode == "origin":
        # 使用原始onnx推理查看下onnx是否正确
        infer_origin_onnx(
            onnx_model_path=onnx_model_path,
            image_path=image_path,
            result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
            height=args.height,
            width=args.width,
        )
    elif args.mode == "calib":
        infer_calib_onnx(
            onnx_model_path=onnx_model_path,
            image_path=image_path,
            result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
            height=args.height,
            width=args.width,
        )
    elif args.mode == "quant":
        infer_quant_onnx(
            onnx_model_path=onnx_model_path,
            image_path=image_path,
            result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
            height=args.height,
            width=args.width,
        )