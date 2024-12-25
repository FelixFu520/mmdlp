# 20241124
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241124/con_DOSOD_L_v1.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241124/output_v2/dosod-l_epoch_40_kxj_rep-without-nms_20241124_1024x1024_672x896_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241124/dosod-l_epoch_40_kxj_rep-without-nms_20241124_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241124/output_v2/dosod-l_epoch_40_kxj_rep-without-nms_20241124_1024x1024_672x896_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241124/output_v2/dosod-l_epoch_40_kxj_rep-without-nms_20241124_1024x1024_672x896_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241124/eval_float_v2_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241124/eval_calib_v2_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241124/eval_quant_v2_data20241103 \
    --show_dir eval_result_show_v1_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp/ python /home/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/dosod/20241124/eval_float_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.65  | 0.854  | 0.762  | 0.406 | 0.631 | 0.76  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.61  | 0.758  | 0.735  | 0.451 | 0.612 | 0.681 |
| skein        | 0.641 | 0.875  | 0.743  | 0.239 | 0.546 | 0.745 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/fa.fu/work/mmdlp/ python /home/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/dosod/20241124/eval_calib_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.612 | 0.845  | 0.725  | 0.385 | 0.585 | 0.733 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.593 | 0.784  | 0.742  | 0.436 | 0.579 | 0.677 |
| skein        | 0.609 | 0.852  | 0.715  | 0.154 | 0.48  | 0.733 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

```