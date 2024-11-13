import numpy as np
import argparse
import os

def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度。

    参数：
    a, b -- 输入的numpy数组（一维）

    返回值：
    两个向量的余弦相似度
    """
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate cosine similarity between two vectors.")
    parser.add_argument("--float_npy_path", 
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_float_v3",
                        type=str, required=False, help="Path to the float numpy file.")
    parser.add_argument("--quant_npy_path",
                        default="/home/users/fa.fu/work/work_dirs/dosod/20241103/eval_quant_v3",
                        type=str, required=False, help="Path to the quant numpy file.")
    args = parser.parse_args()

    all_samples = os.listdir(args.float_npy_path)
    for sample in all_samples:
        bbox_float_npy = np.load(os.path.join(args.float_npy_path, sample, "bbox_preds.npy"))
        bbox_quant_npy = np.load(os.path.join(args.quant_npy_path, sample, "bbox_preds.npy"))
        cosine_bbox_value = cosine_similarity(bbox_float_npy.flatten(), bbox_quant_npy.flatten())

        score_float_npy = np.load(os.path.join(args.float_npy_path, sample, "cls_scores.npy"))
        score_quant_npy = np.load(os.path.join(args.quant_npy_path, sample, "cls_scores.npy"))
        cosine_score_value = cosine_similarity(score_float_npy.flatten(), score_quant_npy.flatten())

        if cosine_bbox_value < 0.9:
            print(f"*********Sample: {sample}, Cosine similarity of bbox: {cosine_bbox_value}, Cosine similarity of score: {cosine_score_value}")
        elif cosine_score_value < 0.9:
            print(f"---------Sample: {sample}, Cosine similarity of bbox: {cosine_bbox_value}, Cosine similarity of score: {cosine_score_value}")
        else:
            pass
            # print(f"Sample: {sample}, Cosine similarity of bbox: {cosine_bbox_value}, Cosine similarity of score: {cosine_score_value}")
