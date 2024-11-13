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
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241023.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py \
    --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v1 \
    --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v1 >> cosine.log

```