# 查找mAP掉点问题
1. 思路1: 输入rgb转nv12导致掉点, 排查思路, 将模型转换时yaml文件中的input_rt_type改成featuremap, 以去掉数据转换, 如果是这个原因, 训练端改输入数据
2. 思路2: 某个算子导致掉点, 排查思路, 看下calibration onnx的精度, 做敏感OP分析, 找到掉点OP, 改为cpu, 如果是这个原因, 通知工具链的人

### con_DOSOD_L_v1.yaml
测试个标准, 复现情况
所有节点采用int16;
使用最完整的矫正集;
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241116_debug/con_DOSOD_L_v1.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.999989           0.000031     0.000000     0.007664            
boxes       0.999946           2.039685     0.021815     156.492981

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_calib_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_calibrated_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v1_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v1_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v1_data20241103 \
    --show_dir eval_result_show_v1_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v1_data20241114 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v1_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v1_data20241114 \
    --show_dir eval_result_show_v1_data20241114 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.647 | 0.853  | 0.766  | 0.372 | 0.634 | 0.74  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.666 | 0.838  | 0.838  | 0.536 | 0.67  | 0.703 |
| skein        | 0.617 | 0.812  | 0.689  | 0.327 | 0.484 | 0.759 |
| solid stain  | 0.064 | 0.064  | 0.064  | nan   | 0.252 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.615 | 0.854  | 0.732  | 0.377 | 0.591 | 0.724 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.635 | 0.835  | 0.778  | 0.541 | 0.642 | 0.657 |
| skein        | 0.593 | 0.789  | 0.675  | 0.255 | 0.477 | 0.739 |
| solid stain  | 0.039 | 0.043  | 0.043  | nan   | 0.151 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v1_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.61  | 0.854  | 0.734  | 0.351 | 0.58  | 0.733 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.647 | 0.856  | 0.78   | 0.496 | 0.665 | 0.664 |
| skein        | 0.589 | 0.788  | 0.68   | 0.242 | 0.47  | 0.737 |
| solid stain  | 0.043 | 0.043  | 0.043  | nan   | 0.168 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
```
### con_DOSOD_L_v2.yaml
部分修改为int8, 看看int8和int16差多少

```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241116_debug/con_DOSOD_L_v2.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.999987           0.000031     0.000000     0.008305            
boxes       0.999858           3.951316     0.035138     141.513382

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v2/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v2_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v2/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v2_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v2_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v2_data20241103 \
    --show_dir eval_result_show_v2_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v2/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v2_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v2/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v2_data20241114 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v2_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v2_data20241114 \
    --show_dir eval_result_show_v2_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.647 | 0.853  | 0.766  | 0.372 | 0.634 | 0.74  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.666 | 0.838  | 0.838  | 0.536 | 0.67  | 0.703 |
| skein        | 0.617 | 0.812  | 0.689  | 0.327 | 0.484 | 0.759 |
| solid stain  | 0.064 | 0.064  | 0.064  | nan   | 0.252 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.612 | 0.847  | 0.722  | 0.373 | 0.591 | 0.728 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.65  | 0.857  | 0.813  | 0.466 | 0.671 | 0.696 |
| skein        | 0.589 | 0.792  | 0.673  | 0.248 | 0.465 | 0.735 |
| solid stain  | 0.026 | 0.026  | 0.026  | nan   | 0.101 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.598 | 0.847  | 0.71   | 0.36  | 0.57  | 0.72  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.654 | 0.864  | 0.795  | 0.493 | 0.684 | 0.656 |
| skein        | 0.583 | 0.795  | 0.668  | 0.27  | 0.458 | 0.729 |
| solid stain  | 0.064 | 0.064  | 0.064  | nan   | 0.252 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

```
### con_DOSOD_L_v3.yaml
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data_featuremap.py \
    --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1115_all \
    --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1115_all_rgb_672×896-featuremap \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241116_debug/con_DOSOD_L_v3.yaml --model-type onnx


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --image_path /home/users/fa.fu/work/work_dirs/dosod/demo_images/0892.jpg \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_featuremap.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v3_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v3_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v3_data20241103 \
    --show_dir eval_result_show_v3_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_featuremap.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v3/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v3_data20241114 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v3_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v3_data20241114 \
    --show_dir eval_result_show_v3_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v3_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.647 | 0.853  | 0.766  | 0.372 | 0.634 | 0.74  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.666 | 0.838  | 0.838  | 0.536 | 0.67  | 0.703 |
| skein        | 0.617 | 0.812  | 0.689  | 0.327 | 0.484 | 0.759 |
| solid stain  | 0.064 | 0.064  | 0.064  | nan   | 0.252 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v3_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.613 | 0.852  | 0.73   | 0.379 | 0.591 | 0.722 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.646 | 0.859  | 0.813  | 0.528 | 0.659 | 0.663 |
| skein        | 0.588 | 0.791  | 0.674  | 0.256 | 0.459 | 0.737 |
| solid stain  | 0.029 | 0.032  | 0.032  | nan   | 0.114 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v3_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.608 | 0.848  | 0.707  | 0.365 | 0.579 | 0.731 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.647 | 0.868  | 0.792  | 0.492 | 0.669 | 0.657 |
| skein        | 0.584 | 0.791  | 0.674  | 0.24  | 0.464 | 0.73  |
| solid stain  | 0.058 | 0.064  | 0.064  | nan   | 0.227 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

```

### con_DOSOD_L_v4.yaml
```
hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241116_debug/con_DOSOD_L_v4.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_featuremap.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v4_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v4_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v4_data20241103 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v4_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v4_data20241103 \
    --show_dir eval_result_show_v4_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_featuremap.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896.onnx \
    --onnx_calib_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v4_calibrated_model.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v4_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v4_data20241114 \
    --save_dir_calib /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v4_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v4_data20241114 \
    --show_dir eval_result_show_v4_data20241114 \
    --height 672 \
    --width 896




PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v4_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.647 | 0.853  | 0.766  | 0.372 | 0.634 | 0.74  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.666 | 0.838  | 0.838  | 0.536 | 0.67  | 0.703 |
| skein        | 0.617 | 0.812  | 0.689  | 0.327 | 0.484 | 0.759 |
| solid stain  | 0.064 | 0.064  | 0.064  | nan   | 0.252 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v4_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.618 | 0.852  | 0.732  | 0.366 | 0.592 | 0.73  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.634 | 0.831  | 0.771  | 0.541 | 0.633 | 0.673 |
| skein        | 0.595 | 0.792  | 0.683  | 0.274 | 0.477 | 0.739 |
| solid stain  | 0.039 | 0.043  | 0.043  | nan   | 0.151 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_quant_v4_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.615 | 0.852  | 0.718  | 0.364 | 0.587 | 0.728 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.632 | 0.831  | 0.772  | 0.541 | 0.628 | 0.676 |
| skein        | 0.593 | 0.789  | 0.683  | 0.261 | 0.476 | 0.738 |
| solid stain  | 0.039 | 0.043  | 0.043  | nan   | 0.151 | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/compare_cos.py \
    --float_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_float_v4_data20241103 \
    --quant_npy_path /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/eval_calib_v4_data20241103


```

### con_DOSOD_L_v5.yaml
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/get_sensitivity_of_nodes.py \
    --model_file /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241115_1024x1024_672x896_v4_calibrated_model.onnx \
    --output_dir /home/users/fa.fu/work/work_dirs/dosod/20241116_debug/output-v4-debug/ \
    --calibrated_data /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1115_all_rgb_672×896-featuremap/
```