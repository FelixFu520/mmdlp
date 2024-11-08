import os
import cv2
import numpy as np
from numpy import ndarray
import os.path as osp
from matplotlib.figure import Figure
from horizon_tc_ui.data.transformer import *

classes = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
palette = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]
def set_image(image: np.ndarray) -> None:
    """Set the image to draw.

    Args:
        image (np.ndarray): The image to draw.
    """
    assert image is not None
    image = image.astype("uint8")
    width, height = image.shape[1], image.shape[0]

    fig_save = Figure(frameon=False)
    ax_save = fig_save.add_subplot()
    ax_save.axis(False)

    # remove white edges by set subplot margin
    fig_save.subplots_adjust(left=0, right=1, bottom=0, top=1)
    dpi = fig_save.get_dpi()
    # add a small 1e-2 to avoid precision lost due to matplotlib's
    # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
    fig_save.set_size_inches(  # type: ignore
        (width + 1e-2) / dpi, (height + 1e-2) / dpi
    )
    ax_save.cla()
    ax_save.axis(False)
    ax_save.imshow(image, extent=(0, width, height, 0), interpolation="none")
def imwrite(
    img: np.ndarray,
    file_path: str,
) -> bool:

    img_ext = osp.splitext(file_path)[-1]
    flag, img_buff = cv2.imencode(img_ext, img)
    # fileio.put(img_buff.tobytes(), file_path, backend_args=None)
    if flag:
        # 将 img_buff 转换为字节流并写入文件
        with open(file_path, 'wb') as f:
            f.write(img_buff.tobytes())
    else:
        print("Failed to encode the image.")

    return flag
def draw_sem_seg(sem_seg, classes, palette, image, alpha=0.5, out_file=None):
    num_classes = len(classes)
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    mask = np.zeros_like(image, dtype=np.uint8)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color

    color_seg = (image * (1 - alpha) + mask * alpha).astype(np.uint8)
    set_image(color_seg)
    drawn_img = color_seg[..., ::-1]
    imwrite(drawn_img, out_file)
    return color_seg


def postprocess(
        outputs: ndarray, 
        width: int = 1024, 
        height:int = 512, 
        result_dir:str = None, 
        image_path:str = None, 
    ) -> None:
    output = outputs[0]
    softmax_output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
    label = np.argmax(softmax_output, axis=1).astype(np.uint8).squeeze(axis=0)
    label = cv2.resize(label, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    scores = np.max(softmax_output, axis=1).squeeze(axis=0)
    scores = cv2.resize(scores, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        # 保存labelids
        cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_labelids.png")), label)
        # 保存scores
        cv2.imwrite(os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_scores.png")), (scores*255).astype(np.uint8))
        # 保存color_wrapper
        draw_sem_seg(label, classes, palette, image=cv2.resize(cv2.imread(image_path), (width, height)), alpha=0.5, 
                        out_file=os.path.join(result_dir, os.path.basename(image_path[:-4] + f"_color.png")))
