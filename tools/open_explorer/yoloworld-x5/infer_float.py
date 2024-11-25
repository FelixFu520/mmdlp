# 在mm环境下
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import PIL.Image
import cv2
import supervision as sv
import os
import json
import argparse

def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0]) # type: ignore
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"

bounding_box_annotator = sv.BoxAnnotator() # type: ignore
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER) # type: ignore
mask_annotator = sv.MaskAnnotator() # type: ignore

open_class_names = ("person,head,hand,arm,body,leg,foot,whiteboard,keyboard,mouse,laptop,marker pen,cup,bottle,eraser,microphone,mobile phone")

def run_image(
        runner,
        input_image,
        max_num_boxes=100,
        score_thr=0.21,
        nms_thr=0.5,
        output_image="output.png",
        out_dir = None,
):    
    
    image_name = os.path.basename(input_image)

    # 开放词测试 
    texts = [[t.strip()] for t in open_class_names.split(",")]
    print("texts:", len(texts))

    output_image = image_name
    # print("texts[2]:", texts[2])

    os.makedirs(out_dir, exist_ok=True) # type: ignore
    output_image = out_dir+output_image

    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    # predictions
    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None
        
    detections = sv.Detections( # type: ignore
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )

    # label ids with confidence scores
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence) # type: ignore
    ]

    # draw bounding box with label
    image = PIL.Image.open(input_image)
    svimage = np.array(image)
    svimage = bounding_box_annotator.annotate(svimage, detections)
    svimage = label_annotator.annotate(svimage, detections, labels)
    if masks is not None:
        svimage = mask_annotator.annotate(image, detections)

    # save output image
    cv2.imwrite(output_image, svimage[:, :, ::-1])  # type: ignore 
    print(f"Results saved to {colorstr('bold', output_image)}")

    return svimage[:, :, ::-1]  # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="/home/users/fa.fu/work/github/YOLO-World-x5/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py")
    parser.add_argument("--img_path", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/yoloworld-x5/demo_images/meeting_room.jpg")
    parser.add_argument("--out_dir", type=str, 
                        default="/home/users/fa.fu/work/work_dirs/yoloworld-x5/result/")
    parser.add_argument("--load_from", type=str,
                        default="/home/users/fa.fu/work/work_dirs/yoloworld-x5/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth")
    parser.add_argument("--work_dir", type=str,
                        default="/home/users/fa.fu/work/work_dirs/yoloworld-x5/result/")
    args = parser.parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.load_from = args.load_from
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline) # type: ignore

    # run model evaluation
    runner.model.eval()

    img_path = args.img_path

    out_dir = args.out_dir
    img = run_image(runner, img_path, out_dir=out_dir)
    sv.plot_image(img) # type: ignore