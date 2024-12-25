import onnx
import argparse


def get_onnx_nodes(onnx_model_path) -> dict:
    # 加载ONNX模型
    model = onnx.load(onnx_model_path)
    node_dict = {}
    node_dict['Conv'] = []
    node_dict['Other'] = []

    # 遍历模型中的所有节点
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Conv':
            node_dict['Conv'].append(node.name)
            # print(f"Conv节点 {i}:")
            # print(f"  名称: {node.name}")
            # print(f"  操作类型: {node.op_type}")
            # print(f"  输入: {node.input}")
            # print(f"  输出: {node.output}")
            # print(f"  属性: {node.attribute}")
        else:
            node_dict['Other'].append(node.name)
            # print(f"其他节点 {i}:")
            # print(f"  名称: {node.name}")
            # print(f"  操作类型: {node.op_type}")
            # print(f"  输入: {node.input}")
            # print(f"  输出: {node.output}")
            # print(f"  属性: {node.attribute}")
            # print()
    return node_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print ONNX nodes')
    parser.add_argument('--onnx_model_path', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444.onnx', help='ONNX model path')
    parser.add_argument('--output_path', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444.txt', help='Output file path')
    args = parser.parse_args()
    onnx_model_path = args.onnx_model_path
    node_dict = get_onnx_nodes(onnx_model_path)
    print(len(node_dict['Conv']))
    print(len(node_dict['Other']))
    # with open(args.output_path, 'w') as f:
    #     for node in node_dict['Conv']:
    #         f.write("Conv-" + node + '\n')
    #     for node in node_dict['Other']:
    #         f.write("Others-" + node + '\n')