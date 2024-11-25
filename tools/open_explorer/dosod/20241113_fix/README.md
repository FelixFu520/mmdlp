# fix

```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241113_fix/con_DOSOD_L_v1.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113_fix/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_for_shiyuan.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113_fix/output_v1/DOSOD_L_without_nms_v1_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/20241113_fix/reinjection_stain_dataset/reinjection_stain_dataset/real_resize_jpg_data \
    --result_dir /home/users/fa.fu/work/work_dirs/dosod/20241113_fix/reinjection_stain_dataset/reinjection_stain_dataset/result \
    --height 672 \
    --width 896
```