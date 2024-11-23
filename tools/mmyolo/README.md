# mmyolo 学习笔记

## 检测示例
https://mmyolo.readthedocs.io/zh-cn/dev/get_started/15_minutes_object_detection.html#

```
# 1. 训练
python /usr/local/lib/python3.8/dist-packages/mmyolo/.mim/tools/misc/download_dataset.py \
    --dataset-name cat \
    --save-dir /root/data/datasets/cat \
    --unzip \
    --delete

python /usr/local/lib/python3.8/dist-packages/mmyolo/.mim/tools/train.py \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    --work-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat

python /usr/local/lib/python3.8/dist-packages/mmyolo/.mim/tools/train.py \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    --resume

tensorboard --logdir=/root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat


# 2. 验证
python /usr/local/lib/python3.8/dist-packages/mmyolo/.mim/tools/test.py \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --show-dir show_results

python /root/mmdlp/tools/mmyolo/featmap_vis_demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --target-layers backbone \
    --channel-reduction squeeze_mean \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/featmap_vis_output

python /root/mmdlp/tools/mmyolo/featmap_vis_demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --target-layers neck \
    --channel-reduction squeeze_mean \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/featmap_vis_output

python /root/mmdlp/tools/mmyolo/boxam_vis_demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --target-layer neck.out_layers[2] \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/featmap_vis_output_cam

python /root/mmdlp/tools/mmyolo/boxam_vis_demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --target-layer neck.out_layers[1] \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/featmap_vis_output_cam

python /root/mmdlp/tools/mmyolo/boxam_vis_demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --target-layer neck.out_layers[0] \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/featmap_vis_output_cam

# 3. 导出
python /root/mmdlp/tools/mmyolo/projects/easydeploy/tools/export_onnx.py \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/ \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 11 \
    --backend ONNXRUNTIME \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25 \
    --model-only

PYTHONPATH=/root/mmdlp/tools/mmyolo python /root/mmdlp/tools/mmyolo/projects/easydeploy/tools/image-demo.py \
    /root/data/datasets/cat/images/IMG_20221020_112705.jpg \
    /root/mmdlp/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.onnx \
    --device cpu \
    --out-dir /root/data/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/infer_onnx_image.jpg
```