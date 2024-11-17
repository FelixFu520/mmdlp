# 查找mAP掉点问题
1. 思路1: 输入rgb转nv12导致掉点, 排查思路, 将模型转换时yaml文件中的input_rt_type改成featuremap, 以去掉数据转换, 如果是这个原因, 训练端改输入数据
2. 思路2: 某个算子导致掉点, 排查思路, 看下calibration onnx的精度, 做敏感OP分析, 找到掉点OP, 改为cpu, 如果是这个原因, 通知工具链的人

### con_DOSOD_L_v1.yaml
所有节点采用int16; 使用最完整的矫正集;
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241116_debug/con_DOSOD_L_v1.yaml --model-type onnx


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_calib_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_float_v3_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_calib_v3_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_quant_v3_data20241103 \
    --show_dir eval_result_show_v3_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_float_v3_data20241114 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_calib_v3_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116/eval_quant_v3_data20241114 \
    --show_dir eval_result_show_v3_data20241114 \
    --height 672 \
    --width 896

```