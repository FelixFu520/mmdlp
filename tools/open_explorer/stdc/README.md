## run
导出onnx, 并使用horizon的PTQ量化过程
1~2 需要在有mm系列的环境下执行, 3~6 需要在有OE的环境上运行, 7是两个环境都需要使用
```
# 1. 导出onnx
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/configs/stdc_horizon/export.py

# 2. 从训练集或者验证集中筛选矫正数据, 最好是有代表性, 每类都来点就行
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/configs/stdc_horizon/collect_calibration_data.py

# 3. 推理原始onnx, 检验是否准确
PYTHONPATH=/home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc python /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/infer_original_onnx.py

# 4. 准备校准数据集
PYTHONPATH=/home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc python /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/gen_calibration_data.py

# 5. 模型转换
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/stdc1_1024×2048_v1.yaml --model-type onnx

# 6. 推理量化onnx, 与原始onnx对比
PYTHONPATH=/home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc python /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/infer_quantized_onnx.py

# 7. 验证pth, 原始onnx, 量化onnx的指标
# pth验证(需要在mm系列的环境上跑)
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/eval_pth.py /home/users/fa.fu/work/mmdlp/configs/stdc_horizon/config_from_mmseg/stdc/stdc1_in1k-pre_4xb12-80k_cityscapes-512x1024.py --work-dir /home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_pth

# float, quant onnx跑出npy结果, 然后将npy结果在mm环境上跑下指标
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/stdc/eval_onnx.py
```
