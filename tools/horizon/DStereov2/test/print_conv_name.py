import onnx

def print_conv_names(model_path):
    # 加载ONNX模型
    model = onnx.load(model_path)
    # 遍历模型中的所有节点
    for node in model.graph.node:
        # 判断节点的操作类型是否为Conv（卷积操作）
        if node.op_type == "Conv":
            print(node.name)

# 指定你的ONNX模型文件路径
model_path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v1_split/PTQ_check_yuv444_optimized_float_model_split.onnx"
print_conv_names(model_path)