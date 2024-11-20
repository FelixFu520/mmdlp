# yolov8 测试是否掉点

### con_DOSOD_L_v1: int16, 其中有resize变成了int8, softmax跑在CPU上了
dosod-l 模型会在测试集上mAP掉点, 所以用yolov8试下

```
导出onnx, 并使用horizon的PTQ量化过程
1~2 需要在有mm系列的环境下执行, 3~6 需要在有OE的环境上运行, 7是两个环境都需要使用

# 1. 导出onnx
这个导出由@陈世源来做, 所以这里没有代码

# 2. 从训练集或者验证集中筛选矫正数据, 最好是有代表性, 每类都来点就行
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/collect_calibration_data.py \
    --datasets_dir /horizon-bucket/AIoT-data-bucket/AIOT_algorithm_data/train_stain_dataset/real_resize_jpg_data_20241103 \
    --calibrate_data_dir /home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1 \
    --number 140

# 3. 推理原始onnx, 检验是否准确
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/yolov8/20241120/yolov8-l_epoch_40_kxj-without-nms_20241103_1024x1024_672x896.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/yolov8/demo_images/0892.jpg \
    --result_dir /home/users/fa.fu/work/work_dirs/yolov8/result \
    --height 672 \
    --width 896

# 4. 准备校准数据集
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1_rgb_20241120 \
    --height 672 \
    --width 896

# 5. 模型转换
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/20241120/con_DOSOD_L_v1.yaml --model-type onnx

# 6. 推理量化onnx, 与原始onnx对比
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/yolov8/20241120/output-v1/yolov8-l_epoch_40_kxj-without-nms_20241103_1024x1024_672x896_v1_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/yolov8/demo_images/0892.jpg \
    --result_dir /home/users/fa.fu/work/work_dirs/yolov8/result \
    --height 672 \
    --width 896

# 7. 验证pth, 原始onnx, 量化onnx的指标
# pth验证(需要在mm系列的环境上跑)
@陈世源验证, 这里略

# float, quant onnx跑出npy结果, 然后将npy结果在mm环境上跑下指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py

# 用mm 继续推理npy, 获得指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_metrics.py
```

### con_DOSOD_L_v2: 在v1的基础上, 修改input_type_rt为featuremap
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/gen_calibration_data_featuremap.py \
    --data_dir /home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/yolov8/calibration_data/calibration_images_v1_rgb_20241120_featuremap \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/20241120/con_DOSOD_L_v2.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yolov8/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/yolov8/20241120/output-v2/yolov8-l_epoch_40_kxj-without-nms_20241103_1024x1024_672x896_v2_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/yolov8/demo_images/0892.jpg \
    --result_dir /home/users/fa.fu/work/work_dirs/yolov8/result \
    --height 672 \
    --width 896

```