# 20241124
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241124/con_DOSOD_L_v1.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241124/output_v2/dosod-l_epoch_40_kxj_rep-without-nms_20241124_1024x1024_672x896_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
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

```