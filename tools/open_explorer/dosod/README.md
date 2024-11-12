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