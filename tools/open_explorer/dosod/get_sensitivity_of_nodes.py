import logging
import argparse
import os
from horizon_nn.quantizer.debugger import get_sensitivity_of_nodes
# 配置日志
logging.basicConfig(filename='get_sensitivity_of_nodes.log', level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get sensitivity of nodes')
    parser.add_argument('--model_file', type=str, 
                        default='./model_output-v3/DOSOD_L_4_without_nms_int16_nv12_conv_int8_1023_calibrated_model.onnx',
                        help='model or file')
    parser.add_argument("--output_dir", type=str, 
                        default='./model_output-v3/', 
                        help='output directory')
    parser.add_argument("--calibrated_data", type=str,
                        default='./model_output-v3/calibration_data',
                        help='calibrated data')
    args = parser.parse_args()


    model_output = args.output_dir
    calibrated_model = args.model_file
    cali_dataset = args.calibrated_data

    metrics=['cosine-similarity', 'mse', 'mre', 'sqnr', 'chebyshev']
    metrics=['cosine-similarity']
    # 节点敏感度
    node_message = get_sensitivity_of_nodes(
        model_or_file=calibrated_model,
        metrics=metrics,
        calibrated_data=cali_dataset,
        node_type='node',
        verbose=True,
    )

    # 激活敏感度
    node_message = get_sensitivity_of_nodes(
        model_or_file=calibrated_model,
        metrics=metrics,
        calibrated_data=cali_dataset,
        node_type='activation',
        verbose=True,
    )

    # 权重敏感度
    node_message = get_sensitivity_of_nodes(
        model_or_file=calibrated_model,
        metrics=metrics,
        calibrated_data=cali_dataset,
        node_type='weight',
        verbose=True,
    )
    