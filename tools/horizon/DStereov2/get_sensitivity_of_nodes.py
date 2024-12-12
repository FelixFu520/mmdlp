import logging
import argparse
import os
from hmct.quantizer.debugger import get_sensitivity_of_nodes
# 配置日志
logging.basicConfig(filename='get_sensitivity_of_nodes.log', level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get sensitivity of nodes')
    parser.add_argument('--model_output', type=str,
                        default='/home/fa.fu/work/work_dirs/horizon/DStereov2/20241210/output_v4/',
                        help="model or file"
                        )
    parser.add_argument('--onnx_name', type=str, 
                        default='PTQ_check_yuv444_calibrated_model.onnx',
                        help='model or file')
    
    args = parser.parse_args()
    
    model_output = args.model_output
    onnx_name = args.onnx_name

    

    calibrated_model =  model_output + onnx_name
    cali_dataset =  model_output + "calibration_data"

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
    