# DOSOD
这个主要是针对扫地机器人项目所涉及的内容

```

导出onnx, 并使用horizon的PTQ量化过程
1~2 需要在有mm系列的环境下执行, 3~6 需要在有OE的环境上运行, 7是两个环境都需要使用

# 1. 导出onnx
这个导出由@陈世源来做, 所以这里没有代码

# 2. 从训练集或者验证集中筛选矫正数据, 最好是有代表性, 每类都来点就行
矫正数据集由@陈世源提供, 所以这里无需操作

# 3. 推理原始onnx, 检验是否准确
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py

# 4. 准备校准数据集
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py

# 5. 模型转换
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v1.yaml --model-type onnx

# 6. 推理量化onnx, 与原始onnx对比
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py

# 7. 验证pth, 原始onnx, 量化onnx的指标
# pth验证(需要在mm系列的环境上跑)
@陈世源验证, 这里略

# float, quant onnx跑出npy结果, 然后将npy结果在mm环境上跑下指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py

# 用mm 继续推理npy, 获得指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_metrics.py
```

### 实验1-修改矫正数据集
2024.11.12, 跑完上面的内容后, 量化会掉点, 主要是score值降低1%~20%个点, 导致量化onnx检测不出来东西, 猜测是因为模型量化的不太好, 现修改量化数据集, 重新量化
```
# 4. 准备校准数据集
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py --data_dir /home/users/fa.fu/work/work_dirs/dosod/calibration_images_1112 --save_dir /home/users/fa.fu/work/work_dirs/dosod/calibration_data_rgb_1112

# 5. 模型转换
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v1_data2.yaml --model-type onnx

# 7. 原始onnx, 量化onnx的指标
# float, quant onnx跑出npy结果, 然后将npy结果在mm环境上跑下指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241103/output_data2/DOSOD_L_without_nms_v0.1_quantized_model.onnx --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float2 --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant2 --show_dir eval_result_show2

# 用mm 继续推理npy, 获得指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float2
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant2

```
结果表示改变矫正数据集后, 对掉精度没啥影响

### 解决精度下降的问题

---
 
v2 修改所有OP为int16
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v2.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241103/output2/DOSOD_L_without_nms_v0.2_quantized_model.onnx --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v2 --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v2 --show_dir eval_result_show_v2

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v2

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v2
```

---
 
v3在v2的基础上修改校准方法为default
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v2.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241103/output3/DOSOD_L_without_nms_v0.3_quantized_model.onnx --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v3 --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v3 --show_dir eval_result_show_v3

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v3

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v3

```

---
 
计算每个输出的余弦相似度
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v3 --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v3 >> cosine.log

```

---
 
第4次试验, 在v3的基础上删除node_info
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v4.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241103/output4/DOSOD_L_without_nms_v0.4_quantized_model.onnx --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v4 --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v4 --show_dir eval_result_show_v4

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v4

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v4

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v4 --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v4 >> cosine.log
```

---

第5次实验, 实验校准时是否在onnx中加入了减均值除方差的预处理节点, 在v4的基础上, 修改校准方法, 修改校准数据集
```
修改gen_calibration_data.py的第39行 为 image = image / 255
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py --data_dir /home/users/fa.fu/work/work_dirs/dosod/calibration_images_1112 --save_dir /home/users/fa.fu/work/work_dirs/dosod/calibration_data_rgb_1113

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241103/con_DOSOD_L_v5.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241103/dosod-l_epoch_40_kxj_rep-without-nms_20241103.onnx --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241103/output5/DOSOD_L_without_nms_v0.5_quantized_model.onnx --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v5 --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v5 --show_dir eval_result_show_v5

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v5

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ --ann_file real_resize_coco_jpg_20241103.json --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v5

结果表明, 减均值除方差的预处理节点在校准时已经加到onnx中了, 所以校准数据不要再减均值除方差了
```


### 20241113 为现场提供模型
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241023_672x896.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb 


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241113/con_DOSOD_L_v1.yaml --model-type onnx


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241023 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241023_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1 \
    --show_dir eval_result_show_v1 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241023.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241023.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py \
    --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1 \
    --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1 >> cosine.log

```

### 20241114 为现场提供模型
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --height 672 \
    --width 896


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241114/con_DOSOD_L_v1.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.804521           0.000032     0.000000     0.008317            
boxes       0.999780           4.396271     0.043458     283.943115


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v1 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v1 \
    --show_dir eval_result_show_v1 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v1 \
    --height 672 \
    --width 896

+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.641 | 0.857  | 0.771  | 0.367 | 0.63  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.826  | 0.785  | 0.508 | 0.666 | 0.677 |
| skein        | 0.622 | 0.84   | 0.71   | 0.389 | 0.5   | 0.744 |
| solid stain  | 0.011 | 0.016  | 0.016  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v1 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.61  | 0.851  | 0.713  | 0.372 | 0.58  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.651 | 0.845  | 0.805  | 0.538 | 0.652 | 0.693 |
| skein        | 0.588 | 0.803  | 0.682  | 0.262 | 0.475 | 0.728 |
| solid stain  | 0.007 | 0.012  | 0.012  | nan   | 0.0   | 0.025 |
+--------------+-------+--------+--------+-------+-------+-------+



PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py \
    --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1 \
    --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1 >> cosine.log

```

### 20241114 为现场提供模型(修改校准数据, 然后再测试, 主要测试校准数据对指标的影响)
```

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_rgb_672×896 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1112 \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1112_rgb_672×896 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241114_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241114_v1_rgb_672×896 \
    --height 672 \
    --width 896


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241114/con_DOSOD_L_v2.yaml --model-type onnx
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241114/con_DOSOD_L_v3.yaml --model-type onnx
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241114/con_DOSOD_L_v4.yaml --model-type onnx


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v3/DOSOD_L_without_nms_v3_quantized_model.onnx \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v4/DOSOD_L_without_nms_v4_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v2 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v2 \
    --show_dir eval_result_show_v2 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v3/DOSOD_L_without_nms_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v3 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v3 \
    --show_dir eval_result_show_v3 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v4/DOSOD_L_without_nms_v4_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v4 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v4 \
    --show_dir eval_result_show_v4 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v2 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v2 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v3 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v3 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v4 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v4 \
    --height 672 \
    --width 896


一、使用校准数据1 mAP “/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb”
数据来自上面一小节

校准数据下的余弦相似度
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.804521           0.000032     0.000000     0.008317            
boxes       0.999780           4.396271     0.043458     283.943115

float onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.641 | 0.857  | 0.771  | 0.367 | 0.63  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.826  | 0.785  | 0.508 | 0.666 | 0.677 |
| skein        | 0.622 | 0.84   | 0.71   | 0.389 | 0.5   | 0.744 |
| solid stain  | 0.011 | 0.016  | 0.016  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+

quant onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.61  | 0.851  | 0.713  | 0.372 | 0.58  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.651 | 0.845  | 0.805  | 0.538 | 0.652 | 0.693 |
| skein        | 0.588 | 0.803  | 0.682  | 0.262 | 0.475 | 0.728 |
| solid stain  | 0.007 | 0.012  | 0.012  | nan   | 0.0   | 0.025 |
+--------------+-------+--------+--------+-------+-------+-------+


二、使用校准数据2 mAP “/home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_rgb_672×896”

校准数据下的余弦相似度
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.999994           0.000031     0.000000     0.004045            
boxes       0.999886           3.459982     0.031280     134.885406

float onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.641 | 0.857  | 0.771  | 0.367 | 0.63  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.826  | 0.785  | 0.508 | 0.666 | 0.677 |
| skein        | 0.622 | 0.84   | 0.71   | 0.389 | 0.5   | 0.744 |
| solid stain  | 0.011 | 0.016  | 0.016  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+

quant onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.609 | 0.853  | 0.707  | 0.364 | 0.583 | 0.727 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.638 | 0.843  | 0.811  | 0.51  | 0.644 | 0.674 |
| skein        | 0.584 | 0.811  | 0.663  | 0.266 | 0.472 | 0.719 |
| solid stain  | 0.005 | 0.006  | 0.006  | nan   | 0.0   | 0.024 |
+--------------+-------+--------+--------+-------+-------+-------+



三、使用校准数据3 mAP “/home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1112_rgb_672×896”

校准数据下的余弦相似度
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.844903           0.000078     0.000026     0.935470            
boxes       0.999871           3.297333     0.033330     231.738922

float onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.641 | 0.857  | 0.771  | 0.367 | 0.63  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.826  | 0.785  | 0.508 | 0.666 | 0.677 |
| skein        | 0.622 | 0.84   | 0.71   | 0.389 | 0.5   | 0.744 |
| solid stain  | 0.011 | 0.016  | 0.016  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+

quant onnx mAP
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.608 | 0.853  | 0.707  | 0.352 | 0.58  | 0.73  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.646 | 0.844  | 0.811  | 0.517 | 0.646 | 0.708 |
| skein        | 0.577 | 0.805  | 0.654  | 0.263 | 0.465 | 0.714 |
| solid stain  | 0.003 | 0.005  | 0.005  | nan   | 0.0   | 0.024 |
+--------------+-------+--------+--------+-------+-------+-------+

三、使用校准数据4 mAP “/home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241114_v1_rgb_672×896”
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.468509           0.000031     0.000000     0.000574            
boxes       0.999620           5.491628     0.057109     274.541931

+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.641 | 0.857  | 0.771  | 0.367 | 0.63  | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.826  | 0.785  | 0.508 | 0.666 | 0.677 |
| skein        | 0.622 | 0.84   | 0.71   | 0.389 | 0.5   | 0.744 |
| solid stain  | 0.011 | 0.016  | 0.016  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+

+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.615 | 0.858  | 0.714  | 0.373 | 0.595 | 0.726 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.649 | 0.846  | 0.811  | 0.572 | 0.647 | 0.699 |
| skein        | 0.586 | 0.807  | 0.675  | 0.276 | 0.47  | 0.724 |
| solid stain  | 0.006 | 0.009  | 0.009  | nan   | 0.0   | 0.039 |
+--------------+-------+--------+--------+-------+-------+-------+
```

### 1023模型再1103测试集 流程
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023.onnx \
    --height 640 \
    --width 640
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023_672x896.onnx \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb_640×640 \
    --height 640 \
    --width 640
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1 \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/20241113_v1_rgb_672×896 \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241023/con_DOSOD_L_v1.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.680810           0.000043     0.000002     0.031314            
boxes       0.999740           3.691333     0.046367     262.430237

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241023/con_DOSOD_L_v2.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.980218           0.000053     0.000005     0.153588            
boxes       0.999864           3.318530     0.034131     261.295288

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --height 640 \
    --width 640
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v1 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v1 \
    --show_dir eval_result_show_v1 \
    --height 640 \
    --width 640
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v2 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v2 \
    --show_dir eval_result_show_v2 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v1 \
    --height 640 \
    --width 640
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.492 | 0.665  | 0.579  | 0.366 | 0.49  | 0.609 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.626 | 0.772  | 0.737  | 0.434 | 0.654 | 0.67  |
| skein        | 0.519 | 0.68   | 0.584  | 0.106 | 0.392 | 0.684 |
| solid stain  | 0.007 | 0.009  | 0.005  | nan   | 0.032 | 0.007 |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v1 \
    --height 640 \
    --width 640
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.455 | 0.636  | 0.551  | 0.311 | 0.446 | 0.585 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.577 | 0.757  | 0.724  | 0.385 | 0.591 | 0.647 |
| skein        | 0.468 | 0.63   | 0.543  | 0.02  | 0.316 | 0.648 |
| solid stain  | 0.007 | 0.009  | 0.006  | nan   | 0.028 | 0.007 |
+--------------+-------+--------+--------+-------+-------+-------+


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v2 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.545 | 0.734  | 0.65   | 0.334 | 0.538 | 0.651 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.631 | 0.787  | 0.75   | 0.475 | 0.654 | 0.662 |
| skein        | 0.504 | 0.68   | 0.559  | 0.13  | 0.42  | 0.649 |
| solid stain  | 0.007 | 0.007  | 0.007  | nan   | 0.023 | 0.017 |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v2 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.505 | 0.705  | 0.59   | 0.355 | 0.493 | 0.627 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.604 | 0.788  | 0.769  | 0.479 | 0.61  | 0.644 |
| skein        | 0.455 | 0.637  | 0.517  | 0.072 | 0.329 | 0.624 |
| solid stain  | 0.006 | 0.007  | 0.007  | nan   | 0.018 | 0.014 |
+--------------+-------+--------+--------+-------+-------+-------+
```

### DOSOD-L blur
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896.onnx \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241114/con_DOSOD_L_v5.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.513604           0.000031     0.000000     0.000545            
boxes       0.999697           5.372114     0.051045     165.431976

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v5/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896_quantized_model.onnx \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v5/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v5 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v5 \
    --show_dir eval_result_show_v5 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v5 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.651 | 0.866  | 0.76   | 0.416 | 0.639 | 0.749 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.614 | 0.79   | 0.774  | 0.431 | 0.649 | 0.657 |
| skein        | 0.639 | 0.857  | 0.735  | 0.404 | 0.502 | 0.758 |
| solid stain  | 0.005 | 0.006  | 0.006  | nan   | 0.0   | 0.022 |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v5 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.61  | 0.852  | 0.693  | 0.358 | 0.578 | 0.734 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.602 | 0.806  | 0.758  | 0.379 | 0.63  | 0.66  |
| skein        | 0.606 | 0.839  | 0.738  | 0.363 | 0.461 | 0.738 |
| solid stain  | 0.01  | 0.011  | 0.011  | nan   | 0.0   | 0.025 |
+--------------+-------+--------+--------+-------+-------+-------+
```


### 1023_640×640, 1023_672×896在1114评测集上评测
```
1023_640×640, 1023_672×896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v1_20241114data \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v1_20241114data \
    --show_dir eval_result_show_v1_20241114data \
    --height 640 \
    --width 640
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241023/dosod-l_epoch_40_kxj_rep-without-nms_20241023_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241023/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v2_20241114data \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v2_20241114data \
    --show_dir eval_result_show_v2_20241114data \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v1_20241114data \
    --height 640 \
    --width 640
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.331 | 0.497  | 0.379  | 0.183 | 0.333 | 0.48  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.173 | 0.316  | 0.168  | 0.0   | 0.022 | 0.343 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v1_20241114data \
    --height 640 \
    --width 640
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.32  | 0.502  | 0.379  | 0.164 | 0.328 | 0.476 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.155 | 0.29   | 0.114  | 0.0   | 0.008 | 0.31  |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_float_v2_20241114data \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.379 | 0.568  | 0.417  | 0.212 | 0.347 | 0.535 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.134 | 0.284  | 0.07   | 0.0   | 0.067 | 0.241 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241023/eval_quant_v2_20241114data \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.359 | 0.548  | 0.386  | 0.272 | 0.287 | 0.548 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.121 | 0.274  | 0.067  | 0.0   | 0.043 | 0.229 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

```
### 1113_672×896_origin_1024×1024在1114评测集上评测
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v1_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v1_data20241114 \
    --show_dir eval_result_show_v1_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v1_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.599 | 0.927  | 0.668  | 0.669 | 0.569 | 0.65  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.441 | 0.697  | 0.488  | 0.0   | 0.296 | 0.625 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v1_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.573 | 0.912  | 0.661  | 0.614 | 0.518 | 0.656 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.385 | 0.633  | 0.434  | 0.0   | 0.224 | 0.595 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

```
### 1113_672×896_origin_1024×1024-blur在1114评测集上评测
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241114/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241114/output_v5/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_672x896_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v5_20241114data \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v5_20241114data \
    --show_dir eval_result_show_v5_20241114data \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v5_20241114data \
    --height 672 \
    --width 896
+--------------+------+--------+--------+-------+-------+-------+
| category     | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+------+--------+--------+-------+-------+-------+
| liquid stain | 0.61 | 0.924  | 0.642  | 0.628 | 0.582 | 0.66  |
| congee stain | nan  | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan  | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.44 | 0.709  | 0.474  | 0.0   | 0.273 | 0.633 |
| solid stain  | nan  | nan    | nan    | nan   | nan   | nan   |
+--------------+------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v5_20241114data \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.558 | 0.885  | 0.613  | 0.567 | 0.492 | 0.641 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.431 | 0.668  | 0.541  | 0.0   | 0.258 | 0.638 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_float_v5_20241114data --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241114/eval_quant_v5_20241114data >> cosine.log
```


以下为全新的内容, 每个中必须评测
