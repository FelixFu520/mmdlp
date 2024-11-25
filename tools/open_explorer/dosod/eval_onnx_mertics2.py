from mmdet.datasets.transforms import PackDetInputs
from mmcv.transforms import LoadAnnotations, LoadImageFromFile
from mmyolo.datasets.transforms import YOLOv5KeepRatioResize, LetterResize
from yolo_world.datasets.transformers.mm_transforms import LoadText
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from yolo_world.datasets.mm_dataset import MultiModalDataset
from mmdet.evaluation import CocoMetric
from mmengine.structures import InstanceData
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
# from mmcv.ops import batched_nms
from torchvision.ops import nms, batched_nms
import argparse
from typing import List, Optional, Tuple
from tqdm import tqdm
import os
import cv2
import json
import numpy as np
import torch
from mmyolo.registry import DATASETS
from mmengine.registry import EVALUATOR, METRICS
from mmengine.evaluator import Evaluator
from mmyolo.utils import register_all_modules
from mmdet.utils import register_all_modules as mmdet_register_all_modules
from yolo_world.datasets import *
from yolo_world.models import *
from yolo_world.engine import *
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmdet.structures import DetDataSample

mmdet_register_all_modules()
register_all_modules(True)

data_root = '/home/users/fa.fu/work/data/dosod_eval_dataset'
class_text_path = "/home/users/fa.fu/work/data/dosod_eval_dataset/kxj_class_texts_1021.json"
ann_file = "real_resize_coco_jpg_20241103.json"
npy_dir = "/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v4"
device = "cuda:4"

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=class_text_path,
        dataset=dict(
            ann_file=f"{data_root}/{ann_file}",
            data_prefix=dict(img=''),
            data_root=data_root,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'liquid stain',
                    'congee stain',
                    'milk stain',
                    'skein',
                    'solid stain',
                )),
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(scale=(640,640), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=f"{data_root}/{ann_file}",
        _scope_='mmdet',
        metric='bbox',
        format_only=False,
        classwise=True,
        metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100'])
]


if __name__ == "__main__":
    evaluator = Evaluator(val_evaluator)      
    dataset = DATASETS.build(val_dataloader['dataset'])
    evaluator.dataset_meta = dataset.metainfo
    
    for data in dataset:
        # 读取真实数据
        inputs = data['inputs']
        data_samples = data['data_samples']
        # print(f"image:{data_samples.get('img_path')}, gt_bbox:{len(data_samples.get('gt_instances')['bboxes'])}")
        image_numpy = inputs.numpy()
        image_numpy = image_numpy.transpose(1, 2, 0)
        image_numpy = np.ascontiguousarray(image_numpy)
        if data_samples.get('img_id') == 13:
            for box in data_samples.get('gt_instances')['bboxes']:
                _box = box.numpy()
                for i in _box:
                    cv2.rectangle(image_numpy, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 2)
            cv2.imwrite(f"/home/users/fa.fu/work/{os.path.basename(data_samples.get('img_path'))}", image_numpy) 

    for data in dataset:
        # 读取真实数据
        inputs = data['inputs']
        data_samples = data['data_samples']
        
        # 读取onnx预测的npy文件
        image_name = os.path.basename(data_samples.get('img_path'))[:-4]
        npy_bbox = np.load(f"{npy_dir}/{image_name}/bbox_preds.npy")
        npy_score = np.load(f"{npy_dir}/{image_name}/cls_scores.npy")
        bboxes = torch.from_numpy(npy_bbox).to(device)
        scores = torch.from_numpy(npy_score).to(device)

        bboxes = bboxes[0]
        scores = scores[0]
        scores, labels = scores.max(1, keepdim=True)
        scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, 0.001, 30000)
        pred_instances = InstanceData(scores=scores,
                                      labels=labels,
                                      bboxes=bboxes[keep_idxs])
        outputs = [DetDataSample(
            pred_instances=pred_instances,
            img_id=data_samples.get('img_id'),
            ori_shape = data_samples.get('ori_shape'),
            instances=data_samples.get('gt_instances'),
            )]
        evaluator.process(data_samples=outputs, data_batch=data)

    metrics = evaluator.evaluate(len(dataset))
    print(metrics)
    print("DONE")
        
