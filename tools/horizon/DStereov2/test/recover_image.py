import numpy as np
from horizon_tc_ui.data.transformer import *
import cv2

left = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_left.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
left_bgr = cv2.cvtColor(left[0], cv2.COLOR_YUV2BGR)
cv2.imwrite("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/left.jpg", left_bgr)

right = np.fromfile("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/onnxcheck_right.npy", dtype=np.uint8).reshape(1, 352, 640, 3)
right_bgr = cv2.cvtColor(right[0], cv2.COLOR_YUV2BGR)
cv2.imwrite("/home/fa.fu/work/mmdlp/tools/horizon/DStereov2/test/right.jpg", right_bgr)