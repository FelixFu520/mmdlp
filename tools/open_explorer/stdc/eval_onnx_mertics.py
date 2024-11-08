      
import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
import onnxsim

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

import mmengine.fileio as fileio
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.evaluation.metrics import IoUMetric

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
        mask[sem_seg[0] == label, :] = color

    color_seg = (image * (1 - alpha) + mask * alpha).astype(np.uint8)
    set_image(color_seg)
    drawn_img = color_seg[..., ::-1]
    imwrite(drawn_img, out_file)
    return color_seg

def mask_transform(mask):
    mask = _class_to_index(np.array(mask).astype('int32'))
    return torch.LongTensor(np.array(mask).astype('int32'))
    
def _class_to_index(mask):
    # assert the value
    values = np.unique(mask)
    # print("values:", values)
    _key = np.array([-1, -1, -1, -1, -1, -1,
                            -1, -1, 0, 1, -1, -1,
                            2, 3, 4, -1, -1, -1,
                            5, -1, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15,
                            -1, -1, 16, 17, 18])
    _mapping = np.array(range(-1, len(_key) - 1)).astype('int32')

    # print("_mapping:", _mapping)
    for value in values:
        assert (value in _mapping)
    # 获取mask中各像素值对应于_mapping的索引
    index = np.digitize(mask.ravel(), _mapping, right=True)

    # print("index:", index)
    # 依据上述索引index，根据_key，得到对应的mask图
    return _key[index].reshape(mask.shape)

def test_mIoU(model_outputs, datasets_dir, output_dir, plot:bool = False):
    os.makedirs(output_dir, exist_ok=True)

    data_samples = []

    # 遍历所有图片
    val_images_path = datasets_dir + '/leftImg8bit/val'
    val_annotations_path = datasets_dir + '/gtFine/val'
    for city in os.listdir(val_images_path):
        city_path = os.path.join(val_images_path, city)
        city_list = os.listdir(city_path)

        for image_name in tqdm(city_list):
            # 获取图片文件
            image_path = os.path.join(city_path, image_name)    # 获得图片文件
            img = cv2.imread(image_path)
            ori_h, ori_w = img.shape[:2]
            ori_shape = (ori_h, ori_w)

            # 获取模型输出的npy文件
            out_path = os.path.join(model_outputs, image_name.replace(".png", '.npy'))  # 获得输出的npy文件
            outputs = np.load(out_path) # 获得输出的npy文件，shape: (1, 19, 1024, 2048)
            output = torch.from_numpy(outputs)
            output = F.interpolate(
                output, size=ori_shape, scale_factor=None, mode="bilinear", align_corners=False
            )
            sem_seg = output[0].argmax(dim=0, keepdim=True)

            # 获取GT
            label_name = image_name.replace('leftImg8bit', 'gtFine_labelIds')
            labelId_path = os.path.join(val_annotations_path, city, label_name)
            img_label = Image.open(labelId_path) 
            img_label = np.array(img_label)
            mask = mask_transform(img_label)       # mask shape: (H,w)
          
            # mIoU V2
            data_sample = SegDataSample()
            gt_semantic_seg = mask
            gt_semantic_seg = torch.LongTensor(gt_semantic_seg)
            gt_sem_seg_data = dict(data=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

            data_sample = data_sample.to_dict()
            
            data_sample['seg_logits'] = dict(
                data=output[0])
            data_sample['pred_sem_seg'] = dict(
                data=sem_seg[0])
            data_sample['img_path'] = image_path
            data_samples.append(data_sample)

            # break

            if plot:
                img = img[..., ::-1]
                out_file = "result_" + image_name
                img_out_path = os.path.join(output_dir, out_file)

                draw_sem_seg(
                    sem_seg, classes, palette, image=img, alpha=0.5, out_file=img_out_path
                )


    # iou_output_dir = "./iou_metric_png"
    # os.makedirs(iou_output_dir, exist_ok=True)
    iou_output_dir= None
    iou_metric = IoUMetric(ignore_index=-1, 
                            iou_metrics=['mIoU'],
                            output_dir=iou_output_dir)
    iou_metric.dataset_meta = dict(
        classes=classes,
        label_map=dict(),
        reduce_zero_label=False)
    
    iou_metric.process([0] * len(data_samples), data_samples)
    
    res = iou_metric.evaluate(len(data_samples))
    print("res:", res)
    print("aAcc:", res["aAcc"])
    print("mIoU:", res["mIoU"])
    print("mAcc:", res["mAcc"])


if __name__ == "__main__":
    datasets_dir = "/horizon-bucket/aidi_public_data/cityscapes/origin"

    print(f"Float onnx 指标评估")
    float_model_outputs = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_float_output2"
    float_output_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_metrics_output_float"
    test_mIoU(float_model_outputs, datasets_dir, float_output_dir, plot=False)

    print(f"Quant onnx 指标")
    qaunt_model_outputs = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_quant_output2"
    quant_output_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/eval_metrics_output_quant"
    test_mIoU(qaunt_model_outputs, datasets_dir, quant_output_dir, plot=False)

    