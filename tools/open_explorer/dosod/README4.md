# 1119日转换
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path  /home/users/fa.fu/work/work_dirs/dosod/20241119/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/030125.jpg \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241119/con_DOSOD_L_v1.yaml --model-type onnx


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path  /home/users/fa.fu/work/work_dirs/dosod/20241119/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896_v1_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_calib_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241119/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896_v1_calibrated_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241119/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241119/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896_v1_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241119/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_20241118_1024x1024_672x896_v1_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_float_v1_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_calib_v1_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_quant_v1_data20241103 \
    --show_dir eval_result_show_v1_data20241103 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_float_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.648 | 0.865  | 0.783  | 0.429 | 0.632 | 0.747 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.634 | 0.798  | 0.778  | 0.452 | 0.657 | 0.697 |
| skein        | 0.607 | 0.826  | 0.696  | 0.168 | 0.467 | 0.748 |
| solid stain  | 0.054 | 0.086  | 0.021  | nan   | 0.0   | 0.21  |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_calib_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.606 | 0.835  | 0.7    | 0.401 | 0.574 | 0.731 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.633 | 0.834  | 0.778  | 0.436 | 0.645 | 0.695 |
| skein        | 0.57  | 0.783  | 0.655  | 0.104 | 0.406 | 0.74  |
| solid stain  | 0.05  | 0.09   | 0.013  | nan   | 0.0   | 0.202 |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241119/eval_quant_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.599 | 0.838  | 0.693  | 0.375 | 0.557 | 0.74  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.619 | 0.814  | 0.764  | 0.386 | 0.64  | 0.672 |
| skein        | 0.568 | 0.787  | 0.651  | 0.105 | 0.392 | 0.738 |
| solid stain  | 0.055 | 0.097  | 0.016  | nan   | 0.0   | 0.163 |
+--------------+-------+--------+--------+-------+-------+-------+
```