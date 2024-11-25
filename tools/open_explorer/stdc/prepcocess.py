import cv2
import numpy as np

# 前处理函数
def preprocess_image(
        image_path: str, 
        height: int = None, 
        width: int = None, 
        bgr_to_rgb: bool = True, 
        to_float: bool = False,
        mean_std: bool = True, 
        MEAN: list = [0.485, 0.456, 0.406],
        STD: list = [0.229, 0.224, 0.225],
        transpose: bool = True,
        new_axis:bool = True, 
    ) -> np.ndarray:
    """
    1. read
    2. gray2bgr
    3. resize
    4. bgr2rgb
    5. to_float
    6. mean_std
    7. transpose
    8. add new_axis
    9. ascontiguousarray
    """
    image = cv2.imread(image_path)

    # 处理灰度图
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 处理resize
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 处理RGB图
    if bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理to_float
    if to_float:
        image = image.astype(np.float32)

    # 处理mean_std
    if mean_std:
        image = image.astype(np.float32)
        image -= MEAN
        image /= STD

    # 转换图像格式
    if transpose:
        image = image.transpose((2, 0, 1))

    # 添加一个新的轴，用于批次维度
    if new_axis:
        image = image[np.newaxis, ...]

    # 确保连续
    image = np.ascontiguousarray(image)  # 确保数据在内存中是连续的

    return image