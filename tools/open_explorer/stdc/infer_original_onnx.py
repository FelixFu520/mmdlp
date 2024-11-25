import os
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime

from prepcocess import preprocess_image
from postprocess import postprocess

MEAN=[123.675, 116.28, 103.53]
STD=[58.395, 57.12, 57.375]

def infer_origin_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height=1024, width=2048):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    image = preprocess_image(image_path, mean_std=True, MEAN=MEAN, STD=STD, bgr_to_rgb=True, transpose=True, height=height, width=width)

    # infer
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)
    
    # postprocess
    postprocess(outputs, width=width, height=height, result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]), image_path=image_path)
   
if __name__ == "__main__":

    onnx_model_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/STDC1_pre.onnx"
    image_path = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/demo_images/krefeld_000000_012353_leftImg8bit.png"
    result_dir = "/home/users/fa.fu/work/work_dirs/stdc_horizon_export_onnx_1024×2048/result"
    os.makedirs(result_dir, exist_ok=True)

    # 使用原始onnx推理查看下onnx是否正确
    infer_origin_onnx(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        result_dir=osp.join(result_dir, osp.basename(onnx_model_path)[:-5]),
    )