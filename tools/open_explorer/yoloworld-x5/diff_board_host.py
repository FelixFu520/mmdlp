import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_score", type=str, default="scores.npy")
    parser.add_argument("--host_bbox", type=str, default="bboxes.npy")
    parser.add_argument("--board_score", type=str, default="model_infer_output_0_scores.bin")
    parser.add_argument("--board_bbox", type=str, default="model_infer_output_1_boxes.bin")
    args = parser.parse_args()

    # quantized.onnx 运行存储的结果
    scores = np.load(args.host_score)
    bboxes = np.load(args.host_bbox)
    print("bboxes:", scores.shape)
    print("bboxes:", bboxes[0, 0, :])

    board_scores = np.fromfile(args.board_score, dtype=np.int16).reshape(1, 8400, -1)[:, :, :17].astype(np.float32)
    board_scores *= 0.00003051804378628731  # 这个参数是 hrt_model_exec model_info --model_file=yolo_world_v2_s_int16_nv12.bin 查询到的
    board_bboxes = np.fromfile(args.board_bbox, dtype=np.int16).reshape(1, 8400, -1)[:, :, :4].astype(np.float32)
    board_bboxes *= 0.025132158771157265
    print("board_scores:", board_scores.shape)
    print("board_bboxes:", board_bboxes.shape, board_bboxes[0, 0, :])

    max_diff = np.max(np.abs(scores - board_scores))
    print("scores max_diff:", max_diff)

    max_diff = np.max(np.abs(bboxes - board_bboxes))
    print("bboxes max_diff:", max_diff)