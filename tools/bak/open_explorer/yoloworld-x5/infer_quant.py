
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *

import os
import cv2
import json 
import torch
import numpy as np
import os.path as osp
import supervision as sv
from torchvision.ops import nms
import argparse

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    # image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)
    


class DictToClass:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

def infer_onnx_with_bbox_decoder(onnx_path, inputs, height=640,width=640, output_dir:str = ""):
    sess = HB_ONNXRuntime(onnx_path)
    input_names = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]
    
    image = inputs[0]
    print("image:", image.shape)

    input_data = RGB2NV12Transformer(data_format="HWC").run_transform(image)
    input_data.tofile(os.path.join(output_dir,"input_data.bin")) # 保存数据, 板端推理测试使用
    input_data = NV12ToYUV444Transformer(target_size=(height,width), 
                                         yuv444_output_layout="HWC").run_transform(input_data)

    print("input_data:", input_data.shape)

    input_data = input_data[np.newaxis, ...]
    input_data -= 128
    input_data = input_data.astype(np.int8)
    # inputs = inputs.transpose(0, 2, 3, 1)
    
    feed_dict = {
        input_names: input_data
    }
    outputs = sess.run(output_names, feed_dict)
    scores, bboxes = outputs
    # 保存cpu上的推理结果, 用于对比板端测试
    np.save(os.path.join(output_dir,"scores.npy"), scores)
    np.save(os.path.join(output_dir,"bboxes.npy"), bboxes)
    return scores, bboxes


def visualize(image, bboxes, labels, scores, texts):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def test_quant_model_infer(onnx_path=None, 
                           texts=None, 
                           input_image = "./img001.jpg", 
                           output_image="img001_result.png",
                           score_thr=0.3,
                           nms_thr=0.7,
                            size=(640, 640),
                            max_dets=300,
                            output_dir=""):
    
    ori_image = cv2.imread(input_image)
    h, w = ori_image.shape[:2]

    inputs, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    
    cls_scores, bbox_preds = infer_onnx_with_bbox_decoder(onnx_path, inputs, output_dir=output_dir)
        
    num_classes = len(texts)
    print("num_classes:", num_classes)

    scores = cls_scores
    bboxes = bbox_preds
    print("scores:", scores.shape)

    ori_scores = torch.from_numpy(scores[0]).to('cpu')
    ori_bboxes = torch.from_numpy(bboxes[0]).to('cpu')

    scores_list = []
    labels_list = []
    bboxes_list = []
    # class-specific NMS
    print("(texts):", len(texts))
    for cls_id in range(len(texts)):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(ori_bboxes, cls_scores, iou_threshold=nms_thr)
        cur_bboxes = ori_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    if len(keep_idxs) > max_dets:
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idxs = sorted_idx[:max_dets]
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

    # Get candidate predict info by num_dets
    scores = scores.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    print("pad_param:", pad_param)

    bboxes /= scale_factor
    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])

    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    img_path = osp.join(output_dir, output_image)
    cv2.imwrite(img_path, image_out)
    return image_out



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, 
                        default="./model_output_yolo_world_v2_s_int16_nv12/yolo_world_v2_s_int16_nv12_quantized_model.onnx")
    parser.add_argument("--input_image", type=str, 
                        default="./meeting_room.jpg")
    parser.add_argument("--score_thr", type=float, default=0.21)
    parser.add_argument("--output_image", type=str, 
                        default="output.jpg")
    parser.add_argument("--output_dir", type=str,
                        default="hb_quant_meeting_room_17.jpg")
    parser.add_argument("--class_names", type=str,
                        default="/home/users/junjun.zhao/github/YOLO-World/data/texts/custom_texts_17.json")
    args = parser.parse_args()

    lvis_v1_class_names_path = args.class_names
    with open(lvis_v1_class_names_path) as f:
        data = json.load(f)
    texts = [x for x in data]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    input_image = args.input_image
    score_thr=args.score_thr
    onnx_path = args.onnx_path
    output_image = args.output_image

    test_quant_model_infer(onnx_path, texts, input_image, output_image, 
                           score_thr=score_thr,output_dir=output_dir)