import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print ONNX nodes')
    parser.add_argument('--source', type=str, default='/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/output_v4/node_int8_ok.txt')
    parser.add_argument('--dst', type=str, default="/home/fa.fu/work/work_dirs/d-robotics/dosod/20241223/output_v4/node_int8_ok.json")
    args = parser.parse_args()

    result_json = {}
    value = {'ON': 'BPU', 'InputType': 'int8'}
    with open(args.source, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("Conv-"):
                line = line.replace("Conv-", "")
                node_info = {line: value}
                result_json.update(node_info)
            elif line.startswith("Others-") and ("HzCalibration" not in line) and ("HZ_PREPROCESS_FOR_images" not in line):
                line = line.replace("Others-", "")
                node_info = {line: value}
                result_json.update(node_info)

    with open(args.dst, 'w') as f:
        json.dump(result_json, f, indent=4)