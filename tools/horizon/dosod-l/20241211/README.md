# yuv444

```
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_image_onnx_yuv444.py\
    --height 672 \
    --width 896 \
    --image_path /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg \
    --result_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/result \
    --mode float \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241211/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444.onnx
    


python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/gen_calibration_data_yuv444.py   \
    --data_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_1205 \
    --save_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_1205_nv12_yuv444_672×896 \
    --height 672 \
    --width 896 \
    --mode nv12

hb_mapper makertbin -c /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/20241211/con_DOSOD_L_v1.yaml --model-type onnx


python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_image_onnx_yuv444.py\
    --height 672 \
    --width 896 \
    --image_path /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg \
    --result_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/result \
    --mode calib \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241211/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444_calibrated_model.onnx
    


python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_image_onnx_yuv444.py\
    --height 672 \
    --width 896 \
    --image_path /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg \
    --result_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/result \
    --mode quant \
    --onnx_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241211/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_20241210_1024x1024_672x896_bgr_nv12_yuv444_quantized_model.onnx

```