import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import torch

def compute_micro_averaging(pred_bboxes, pred_scores, pred_labels, 
                     gt_bboxes, gt_labels, thresholds=(0.1, 0.9, 0.05), 
                     iou_threshold=0.8, num_classes=4):
    """计算每个阈值下的 Micro-Averaged Precision, Recall, 和 F1 Score."""
    f1_scores = []
    tp_all_thresh = np.zeros((len(thresholds), num_classes))
    fp_all_thresh = np.zeros((len(thresholds), num_classes))
    fn_all_thresh = np.zeros((len(thresholds), num_classes))
    
    for idx, thresh in enumerate(thresholds):
        # 过滤预测框
        mask = pred_scores >= thresh
        filtered_bboxes = pred_bboxes[mask]
        filtered_labels = pred_labels[mask]
        # 计算所有类别的 TP, FP, FN
        tp_per_thresh = np.zeros((num_classes))
        fp_per_thresh = np.zeros((num_classes))
        fn_per_thresh = np.zeros((num_classes))
        for cls in range(num_classes):
            tp_per_thresh[cls], fp_per_thresh[cls], fn_per_thresh[cls] = match_bboxes(
                filtered_bboxes[filtered_labels == cls], 
                gt_bboxes[gt_labels == cls], 
                iou_threshold
            )

        tp_all_thresh[idx] = tp_per_thresh
        fp_all_thresh[idx] = fp_per_thresh
        fn_all_thresh[idx] = fn_per_thresh
        
    return tp_all_thresh, fp_all_thresh, fn_all_thresh

def match_bboxes(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """匹配预测框与真实框，计算 TP, FP, FN."""
    if len(gt_bboxes) == 0:
        # 如果没有GT框，所有预测框都是 FP
        return 0, len(pred_bboxes), 0
    
    tp, fp = 0, 0
    matched_gt = set()  # 记录已匹配的GT框

    for pred_bbox in pred_bboxes:
        ious = compute_iou(pred_bbox, gt_bboxes)
        if ious.size == 0 or ious.max() < iou_threshold:
            fp += 1  # 没有匹配到的预测框视为 FP
            continue

        max_iou_idx = np.argmax(ious)
        if max_iou_idx not in matched_gt:
            tp += 1  # 成功匹配，计为 TP
            matched_gt.add(max_iou_idx)
        else:
            fp += 1  # 匹配到已使用的GT框，计为 FP

    fn = len(gt_bboxes) - len(matched_gt)  # 未匹配的 GT 计为 FN
    return tp, fp, fn

def compute_iou(box1, boxes):
    """计算一个预测框与多个GT框的IoU."""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-6)  # 防止除零错误


def load_gt_from_json(gt_json_path):
    # gt_json_path = '/home/users/shiyuan.chen/YOLO-world-dosod-dataset/reinjection_stain_dataset/real_resize_coco_jpg.json'
    
    with open(gt_json_path, 'rb') as f:
        data = json.load(f)
    all_gt_instances = []
    
    for image in data['images']:
        image_id = image['id']
        image_instance = {

        }
        bboxes = []
        labels = []
        for anno in data['annotations']:
            if anno['image_id'] == image_id:
                
                coco_bbox = anno['bbox']
                xmin = coco_bbox[0]
                ymin = coco_bbox[1]
                xmax = coco_bbox[0] + coco_bbox[2]
                ymax = coco_bbox[1] + coco_bbox[3]
                
                

                bboxes.append([xmin, ymin, xmax, ymax])
                # bboxes.append(anno['bbox'])
                labels.append(anno['category_id'])

                # if anno['category_id'] == 0
                #     bboxes.append([xmin, ymin, xmax, ymax])
                #     # bboxes.append(anno['bbox'])
                #     labels.append(anno['category_id'])
                # else:
                #     continue

        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        image_instance = {
            'bboxes': bboxes_tensor,
            'labels': labels_tensor
        }



        all_gt_instances.append(image_instance)
    return all_gt_instances


if __name__ == '__main__':
    import json
    import csv
    pickle_paths = [
        "/home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/pth5.pkl"
        # "/home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p_science.pkl"

    ]
    
    
    name_paths = [
        '1113_1024p_896p',
        ]

    save_dir = '/home/fa.fu/work/work_dirs/horizon/dosod-l/pr'
    for idx_p, pickle_path in enumerate(pickle_paths):
        idx_p_save_dir = os.path.join(save_dir, name_paths[idx_p])
        os.makedirs(idx_p_save_dir, exist_ok=True)
        print(f'Processing {pickle_path}...')
        # 加载预测结果和GT
        pickle_data = pickle.load(open(pickle_path, 'rb'))
        # from IPython import embed; embed()
        all_f1_scores = []
        # thresholds_arange = np.arange(0.05, 0.99, 0.001)  # 阈值从0.1到0.9
        thresholds_arange = np.arange(0.05, 0.99, 0.05)
        # thresholds_arange = np.array([0.5])
        num_classes = 1  
        all_tp = np.zeros((len(pickle_data), len(thresholds_arange), num_classes))
        all_fp = np.zeros((len(pickle_data), len(thresholds_arange), num_classes))
        all_fn = np.zeros((len(pickle_data), len(thresholds_arange), num_classes))
        
        for idx, data in enumerate(pickle_data):
            pred_instances = data['pred_instances']
            
            
            # mask = pred_instances['labels'] == 0
            # print(mask)
            # # 仅保留 `labels` 为 0 的实例
            # filtered_pred_instance = {
            #     'bboxes': pred_instances['bboxes'][mask],
            #     'scores': pred_instances['scores'][mask],
            #     'labels': pred_instances['labels'][mask]
            # }
            # # print(filtered_pred_instance)
            # pred_instances = filtered_pred_instance
            
            gt_instances = data['gt_instances']


            tp_all_thresh, fp_all_thresh, fn_all_thresh = compute_micro_averaging(
                pred_instances['bboxes'].numpy(), 
                pred_instances['scores'].numpy(), 
                pred_instances['labels'].numpy(), 
                gt_instances['bboxes'].numpy(), 
                gt_instances['labels'].numpy(),
                thresholds=thresholds_arange,
                num_classes=num_classes
            )

            all_tp[idx] = tp_all_thresh
            all_fp[idx] = fp_all_thresh
            all_fn[idx] = fn_all_thresh
        
        tp_sum = np.sum(all_tp, axis=0)
        fp_sum = np.sum(all_fp, axis=0)
        fn_sum = np.sum(all_fn, axis=0)
        
        f1_curve = []
        precision_curve = []
        recall_curve = []
        wujian = []
        
        for thresh_idx in range(len(thresholds_arange)):
            tp_sum_micro = 0
            fp_sum_micro = 0
            fn_sum_micro = 0
            for cls_idx in range(num_classes):
                tp_sum_thresh_cls = tp_sum[thresh_idx, cls_idx]
                fp_sum_thresh_cls = fp_sum[thresh_idx, cls_idx]
                fn_sum_thresh_cls = fn_sum[thresh_idx, cls_idx]
                tp_sum_micro += tp_sum_thresh_cls
                fp_sum_micro += fp_sum_thresh_cls
                fn_sum_micro += fn_sum_thresh_cls
            precision = tp_sum_micro / (tp_sum_micro + fp_sum_micro)
            recall = tp_sum_micro / (tp_sum_micro + fn_sum_micro)
            f1 = 2 * precision * recall / (precision + recall)
            
            # print('fp_sum_micro:', fp_sum_micro)
            precision_curve.append(precision)
            recall_curve.append(recall)
            wujian.append(fp_sum_micro)
            f1_curve.append(f1)

        
        # from IPython import embed; embed()
        results_file = os.path.join(idx_p_save_dir, f'pr_in_diff_thres_science_{name_paths[idx_p]}.csv')
        with open(results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Threshold', 'Precision', 'Recall', 'Wujian'])
            for threshold, precision, recall, wujian in zip(thresholds_arange, precision_curve, recall_curve, wujian):
                writer.writerow([threshold, precision, recall, wujian])

            # print(f'Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
        # from IPython import embed; embed()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(thresholds_arange, f1_curve, label='F1 Score')
        plt.scatter([thresholds_arange[np.argmax(f1_curve)]], [np.max(f1_curve)], color='red', zorder=5)
        plt.text(thresholds_arange[np.argmax(f1_curve)], np.max(f1_curve), f'Max F1: {np.max(f1_curve):.2f} @ {thresholds_arange[np.argmax(f1_curve)]:.2f}',
                ha='center', va='bottom', fontsize=9, color='red')
        print(f'Max F1: {np.max(f1_curve):.2f} @ {thresholds_arange[np.argmax(f1_curve)]:.2f}')
        plt.xlabel('Score Threshold')
        plt.ylabel('F1 Score')
        plt.title('Micro-Averaged F1 Score Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(thresholds_arange, precision_curve, label='Precision')
        plt.plot(thresholds_arange, recall_curve, label='Recall')
        plt.xlabel('Score Threshold')
        plt.ylabel('Precision / Recall')
        plt.title('Micro-Averaged Precision/Recall Curve')
        plt.legend()

        f1_curve_name = os.path.splitext(os.path.basename(pickle_path))[0] + '_micro_f1_pr_curve.png'
        os.makedirs(os.path.join(idx_p_save_dir, 'f1_pr_curve_kexuejia'), exist_ok=True)
        plt.savefig(os.path.join(idx_p_save_dir, 'f1_pr_curve_kexuejia',f1_curve_name))
        plt.show()
        plt.close()
