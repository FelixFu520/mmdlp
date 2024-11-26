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
import json
import numpy as np
import torch
import pickle


def bbox_post_process(results: InstanceData,
                    cfg:Optional[dict] ,
                    rescale: bool = False,
                    with_nms: bool = True,
                    img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.min_bbox_size >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            
            # 检查是否启用类别无关的 NMS
            if cfg.class_agnostic:
                # print("cfg.class_agnostic:", cfg.class_agnostic)
                # 类别无关的 NMS，将所有 labels 设置为 0
                labels = torch.zeros_like(results.labels)
            else:
                # 类别相关的 NMS，使用原始的 labels
                labels = results.labels

            keep_idxs = batched_nms(bboxes, results.scores,
                                    # results.labels, 
                                    labels,
                                    iou_threshold=cfg.iou_threshold)
            results = results[keep_idxs]
            results = results[:cfg.max_per_img]
        return results


class DictToClass:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)


def coco_eval(data_dir, ann_file, classes, img_scale, data_prefix, pred_npy_dir, cfg, device = "cuda:0"):

    # Define the data transformations
    transforms = [
        LoadImageFromFile(),
        YOLOv5KeepRatioResize(scale=img_scale),
        LetterResize(scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
        LoadAnnotations(with_bbox=True),
        LoadText(),
        PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                'scale_factor', 'pad_param', 'texts'))
    ]

    # Create the base dataset
    base_dataset = YOLOv5CocoDataset(
        data_root=data_dir,
        ann_file=ann_file,
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=classes
            ),
        test_mode=True,
        pipeline=None,  # Pipeline will be applied in MultiModalDataset
        batch_shapes_cfg=None
    )

    # Wrap the base dataset with MultiModalDataset
    coco_val_dataset = MultiModalDataset(
        dataset=base_dataset,
        class_text_path=os.path.join("/horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/texts", "kxj_class_texts_1021.json"),
        pipeline=transforms
    )

    val_dataloader = coco_val_dataset
    test_dataloader = val_dataloader

    # Initialize the evaluator
    val_evaluator = CocoMetric(
        ann_file=os.path.join(data_dir, ann_file),
        metric='bbox',
        classwise=True
    )

    # Use the same evaluator for testing
    test_evaluator = val_evaluator
    test_evaluator.dataset_meta = dict(classes=classes)

    print("***** test_dataloader:", len(test_dataloader))
    # ***** test_dataloader: 4809 
    cfg = DictToClass(cfg)

    results_list = []
    rescale = True
    with_nms = True

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch_img_metas = batch['data_samples']    
        img_id = batch_img_metas.img_id
        img_path = batch_img_metas.img_path
        img_shape = batch_img_metas.img_shape
        base_name = os.path.basename(img_path).replace(".jpg", "")
        # print("**** base_name:", base_name)
        ori_shape = batch_img_metas.ori_shape
        scale_factor = batch_img_metas.scale_factor
        pad_param = batch_img_metas.pad_param
        onnx_out_dir = os.path.join(pred_npy_dir, base_name) 
        scores_np = np.load(onnx_out_dir + "/cls_scores.npy")
        bboxes_np = np.load(onnx_out_dir + "/bbox_preds.npy")
        flatten_cls_scores = torch.from_numpy(scores_np).to(device)
        flatten_decoded_bboxes = torch.from_numpy(bboxes_np).to(device) 
        
        num_imgs = 1 
        flatten_objectness = [None for _ in range(num_imgs)]
        
        for (bboxes, scores, objectness,
                img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                flatten_objectness, [batch_img_metas]):

            score_thr = cfg.score_thr
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.yolox_style:
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.nms_pre
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                    labels=labels,
                                    bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.yolox_style:
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = bbox_post_process(results=results,
                                        cfg=cfg,
                                        rescale=False,
                                        with_nms=with_nms,
                                        img_meta=img_meta)

            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

             # Create data_sample as a dictionary, and include metainfo
            data_sample = {
                'pred_instances': results,
                'ori_shape': ori_shape,
                'img_shape': img_shape,
                'img_id': img_id
                # Add more metainfo if needed
            }

            results_list.append(data_sample)

        
    # Save the results_list as a pkl file
    gt_anno_path = os.path.join(data_dir, ann_file)
    save_pkl(gt_anno_path, results_list)
    
    print("*** results_list:", len(results_list))
    test_evaluator.process({}, results_list)
    size = len(test_dataloader)
    eval_results = test_evaluator.evaluate(size=size)
    print(eval_results)


def save_pkl(gt_anno_path, pred_results):
    gt_json_data = json.load(open(gt_anno_path, 'r'))
    pkl_list = []
    for idx, pred_result in enumerate(pred_results):
        img_id = pred_result['img_id']
        assert img_id == gt_json_data['images'][idx]['id']
        
        gt_instances = load_img_gt_instance(img_id, gt_json_data)
        
        pred_instances = pred_result['pred_instances']
        bboxes = pred_instances.bboxes.cpu()
        scores = pred_instances.scores.cpu()
        labels = pred_instances.labels.cpu()
        pred_instances = {
            'bboxes': bboxes,
            'scores': scores,
            'labels': labels
        }
        
        pkl_list.append({
            'img_id': img_id,
            'ori_shape': pred_result['ori_shape'],
            'img_shape': pred_result['img_shape'],
            'pred_instances': pred_instances,
            'texts': classes,
            'gt_instances': gt_instances
        })

    with open('test.pkl', 'wb') as f:
        pickle.dump(pkl_list, f)

def load_img_gt_instance(img_id, data):
    '''加载单张图片的gt instance'''
    bboxes = []
    labels = []
    for anno in data['annotations']:
        if anno['image_id'] == img_id:
            
            coco_bbox = anno['bbox']
            xmin = coco_bbox[0]
            ymin = coco_bbox[1]
            xmax = coco_bbox[0] + coco_bbox[2]
            ymax = coco_bbox[1] + coco_bbox[3]
            

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(anno['category_id'])

    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
    gt_instances = {
        'bboxes': bboxes_tensor,
        'labels': labels_tensor
    }
    return gt_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCO Evaluation')
    parser.add_argument('--pred_npy_dir', type=str, default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant", help='The directory of the prediction numpy files')
    parser.add_argument('--data_dir', type=str, default="/home/users/fa.fu/work/data/dosod_eval_dataset/", help='The directory of the dataset')
    parser.add_argument('--ann_file', type=str, default="real_resize_coco_jpg_20241103.json", help='The annotation file of the dataset')
    parser.add_argument("--height", type=int,
                        default=640,
                        help="height")
    parser.add_argument("--width", type=int,
                        default=640,
                        help="width")
    args = parser.parse_args()

    classes = (
                'liquid stain',
                'congee stain',
                'milk stain',
                'skein',
                'solid stain',
    )
    # Define image scale
    img_scale = (args.width, args.height)  

    data_prefix = ''
    # TODO: 修改为 quantized.onnx 跑出来的结果路径 
    # pred_npy_dir = "/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float"
    pred_npy_dir = args.pred_npy_dir

    
    # TODO: Eval 时这里的参数配置
    cfg = {'multi_label': True, 
           'nms_pre': 30000, 
            'score_thr': 0.001, 
            'iou_threshold':  0.4,
            'max_per_img': 300,
            "yolox_style": False,
            "class_agnostic": True,
            "min_bbox_size": -1}
    
    device = "cuda:1"
    
    data_dir = args.data_dir
    ann_file = args.ann_file
    coco_eval(data_dir, ann_file, classes, img_scale, data_prefix, pred_npy_dir, cfg, device)



