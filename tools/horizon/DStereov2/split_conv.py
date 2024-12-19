from copy import deepcopy
import numpy as np

from hmct.common import constant_folding
from hmct.ir import load_model, save_model


def split_conv_nodes(model, conv_names):
    for conv_name in conv_names:
        conv_node = model.graph.node_mappings[conv_name]
        before_node = conv_node.inputs[0].src_op
        conv_weight_value = deepcopy(conv_node.inputs[1].value)
        conv_weight_max = abs(conv_weight_value).max(axis=(1,2,3))
        moded = (conv_weight_max / 127)[:, np.newaxis, np.newaxis, np.newaxis] + 1e-10
        conv_weight_high = np.floor(np.clip(conv_weight_value / moded + 1e-5, -127, 127)) * moded
        conv_weight_low = conv_weight_value - conv_weight_high
        conv_bias_value = conv_node.inputs[2].value if len(conv_node.inputs) == 3 else np.zeros(conv_weight_value.shape[0], np.float32)
        conv1_weight_var = model.graph.create_variable(
            is_param=True,
            value=conv_weight_high,
        )
        conv1_bias_var = conv_node.inputs[2] if len(conv_node.inputs) == 3 else model.graph.create_variable(
            is_param=True,
            value=np.zeros_like(conv_bias_value, np.float32),
        )
        conv1_node = model.graph.create_node(
            op_type="Conv",
            name = conv_node.name + "_split0",
            attributes=conv_node.attributes,
            inputs=[before_node.outputs[0], conv1_weight_var, conv1_bias_var],
            num_outputs=1).insert_after(before_node)
        conv2_weight_var = model.graph.create_variable(
            is_param=True,
            value=conv_weight_low,
        )
        conv2_bias_var = model.graph.create_variable(
            is_param=True,
            value=np.zeros_like(conv_bias_value, np.float32),
        )
        conv2_node = model.graph.create_node(
            op_type="Conv",
            name = conv_node.name + "_split1",
            attributes=conv_node.attributes,
            inputs=[before_node.outputs[0], conv2_weight_var, conv2_bias_var],
            num_outputs=1).insert_after(before_node)
        add1_node = model.graph.create_node(
            op_type="Add",
            inputs=[conv1_node.outputs[0], conv2_node.outputs[0]],
            name=conv_node.name + "_split_add0",
            num_outputs=1).insert_after(conv1_node)
        conv_node.replace_all_uses_with(add1_node)
        if not conv_node.is_used:
            conv_node.destroy()
    model.infer_shapes()
    model.check_validity()
    return model


if __name__ == "__main__":
    model = constant_folding(load_model("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v1/PTQ_check_yuv444_optimized_float_model.onnx"))
    model = split_conv_nodes(model, conv_names=[
        # "/backbone/mod1/mod1.0/Conv",   # onnx::Conv_2747_HzCalibration
        # "/backbone/mod1/mod1.0_1/Conv", # onnx::Conv_2864_HzCalibration
        "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0/Conv",   # onnx::Conv_2750_HzCalibration
        "/get_initdisp/GEMM",   # /get_initdisp/GEMMvariable_2921_conv_weight_HzCalibration
        "/backbone/mod2/mod2.0/head_layer/conv/conv.0/conv.0.0_1/Conv", # onnx::Conv_2867_HzCalibration
        "/refinement/update_block/encoder/convd1/Conv", # refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1/Conv_HzCalibration
        "/refinement/update_block/encoder/convd1_1/Conv",   # refinement.update_block.encoder.convd1.weight_/refinement/update_block/encoder/convd1_1/Conv_HzCalibration
    ])
    save_model(model, "/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v1/PTQ_check_yuv444_optimized_float_model_split.onnx")
