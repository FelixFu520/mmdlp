import numpy as np
import cv2
import os
from horizon_tc_ui.data.transformer import *

# path = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub/infra1/4.npy"

# data = np.fromfile(path, dtype=np.uint8).reshape(3, 352, 640)
# data = np.transpose(data, (1, 2, 0))
# # data = data.astype(np.float32)
# data = cv2.cvtColor(data, cv2.COLOR_YUV2BGR)
# cv2.imwrite("/home/fa.fu/work/mmdlp/t.png", data)


if __name__ == "__main__":
    infra1 = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub/infra1"
    infra2 = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub/infra2"
    dst_infra1 = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub_fix/infra1"
    dst_infra2 = "/home/fa.fu/work/work_dirs/horizon/DStereov2/calibration_data/calibration1208_yuv444_sub_fix/infra2"
    os.makedirs(dst_infra1, exist_ok=True)
    os.makedirs(dst_infra2, exist_ok=True)

    infra1_npy = [os.path.join(infra1, t) for t in os.listdir(infra1)]
    infra2_npy = [os.path.join(infra2, t) for t in os.listdir(infra1)]

    for infra1_p, infra2_p in zip(infra1_npy, infra2_npy):
        assert os.path.basename(infra1_p) == os.path.basename(infra2_p)

        infra1_p_data = np.fromfile(infra1_p, dtype=np.uint8).reshape(3, 352, 640)
        infra1_p_data = infra1_p_data.astype(np.float32)
        infra1_p_data.tofile(os.path.join(dst_infra1, os.path.basename(infra1_p)))

        infra2_p_data = np.fromfile(infra2_p, dtype=np.uint8).reshape(3, 352, 640)
        infra2_p_data = infra2_p_data.astype(np.float32)
        infra2_p_data.tofile(os.path.join(dst_infra2, os.path.basename(infra2_p)))
