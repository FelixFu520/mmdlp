README.md中内容太多了, 续接README.md

--- 
## 1113日, 训练输入672×672, onnx输入672×672 一系列实验

### (1). 1113日, 训练输入672×672, onnx输入672×672, yaml文件v2

```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --height 672 \
    --width 896


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241113/con_DOSOD_L_v2.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.642219           0.000067     0.000011     0.282087            
boxes       0.999683           4.737347     0.052202     301.236420


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v2_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v2_data20241103 \
    --show_dir eval_result_show_v2_data20241103 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v2/DOSOD_L_without_nms_v2_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v2_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v2_data20241114 \
    --show_dir eval_result_show_v2_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.649 | 0.872  | 0.76   | 0.374 | 0.636 | 0.75  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.66  | 0.836  | 0.816  | 0.495 | 0.698 | 0.661 |
| skein        | 0.636 | 0.858  | 0.729  | 0.414 | 0.519 | 0.751 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v2_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.61  | 0.85   | 0.714  | 0.361 | 0.586 | 0.735 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.665 | 0.866  | 0.844  | 0.611 | 0.659 | 0.701 |
| skein        | 0.603 | 0.834  | 0.706  | 0.382 | 0.473 | 0.731 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v2_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.601 | 0.916  | 0.664  | 0.651 | 0.552 | 0.677 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.461 | 0.726  | 0.522  | 0.0   | 0.349 | 0.617 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v2_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.551 | 0.892  | 0.611  | 0.534 | 0.514 | 0.631 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.425 | 0.674  | 0.478  | 0.0   | 0.313 | 0.597 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

```

### (2). 1113日, 训练输入672×672, onnx输入672×672, yaml文件v3(修改所有节点为int16)
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --height 672 \
    --width 896


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241113/con_DOSOD_L_v3.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.780223           0.000066     0.000010     0.263288            
boxes       0.999751           4.061510     0.046244     274.386261


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v3/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v3_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v3/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v3_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v3_data20241103 \
    --show_dir eval_result_show_v3_data20241103 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v3/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v3_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v3_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v3_data20241114 \
    --show_dir eval_result_show_v3_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v3_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.649 | 0.872  | 0.76   | 0.374 | 0.636 | 0.75  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.66  | 0.836  | 0.816  | 0.495 | 0.698 | 0.661 |
| skein        | 0.636 | 0.858  | 0.729  | 0.414 | 0.519 | 0.751 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v3_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.612 | 0.852  | 0.724  | 0.357 | 0.585 | 0.738 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.658 | 0.864  | 0.814  | 0.598 | 0.662 | 0.679 |
| skein        | 0.604 | 0.827  | 0.711  | 0.394 | 0.468 | 0.732 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v3_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.601 | 0.916  | 0.664  | 0.651 | 0.552 | 0.677 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.461 | 0.726  | 0.522  | 0.0   | 0.349 | 0.617 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v3_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.558 | 0.905  | 0.596  | 0.554 | 0.525 | 0.633 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.433 | 0.683  | 0.497  | 0.0   | 0.323 | 0.599 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

```

### (3). 1113日, 训练输入672×672, onnx输入672×672, yaml文件v4(修改所有节点为int16, 同时扩大校准数据集)
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/gen_calibration_data.py \
        --data_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1115_all \
        --save_dir /home/users/fa.fu/work/work_dirs/dosod/caliration_data/calibration_images_1115_all_rgb_672×896 \
        --height 672 \
        --width 896

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_original_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --height 672 \
    --width 896


hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/20241113/con_DOSOD_L_v4.yaml --model-type onnx
=============================================================================
Output      Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------------
scores      0.999985           0.000031     0.000000     0.006507            
boxes       0.999908           2.674561     0.028248     276.812042


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/infer_quantized_onnx.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v4/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v4_quantized_model.onnx \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v4/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v4_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v4_data20241103 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v4_data20241103 \
    --show_dir eval_result_show_v4_data20241103 \
    --height 672 \
    --width 896
PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/real_resize_jpg_data_20241114 \
    --onnx_float_path /home/users/fa.fu/work/work_dirs/dosod/20241113/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896.onnx \
    --onnx_quant_path /home/users/fa.fu/work/work_dirs/dosod/20241113/output_v4/dosod-l_epoch_40_kxj_rep-without-nms_20241113_672x896_672x896_v4_quantized_model.onnx \
    --save_dir_float /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v4_data20241114 \
    --save_dir_quant /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v4_data20241114 \
    --show_dir eval_result_show_v4_data20241114 \
    --height 672 \
    --width 896


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v4_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.649 | 0.872  | 0.76   | 0.374 | 0.636 | 0.75  |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.66  | 0.836  | 0.816  | 0.495 | 0.698 | 0.661 |
| skein        | 0.636 | 0.858  | 0.729  | 0.414 | 0.519 | 0.751 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v4_data20241103 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.613 | 0.855  | 0.709  | 0.348 | 0.586 | 0.743 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | 0.658 | 0.864  | 0.811  | 0.59  | 0.667 | 0.669 |
| skein        | 0.609 | 0.836  | 0.72   | 0.394 | 0.469 | 0.731 |
| solid stain  | 0.0   | 0.0    | 0.0    | nan   | 0.0   | 0.0   |
+--------------+-------+--------+--------+-------+-------+-------+


PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_float_v4_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.601 | 0.916  | 0.664  | 0.651 | 0.552 | 0.677 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.461 | 0.726  | 0.522  | 0.0   | 0.349 | 0.617 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

PYTHONPATH=/home/users/fa.fu/work/mmdlp/ python /home/users/fa.fu/work/mmdlp/tools/open_explorer/dosod/eval_onnx_mertics.py \
    --data_dir /home/users/fa.fu/work/data/dosod_eval_dataset/ \
    --ann_file real_resize_coco_jpg_20241114.json \
    --pred_npy_dir /home/users/fa.fu/work/work_dirs/dosod/20241113/eval_quant_v4_data20241114 \
    --height 672 \
    --width 896
+--------------+-------+--------+--------+-------+-------+-------+
| category     | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+--------------+-------+--------+--------+-------+-------+-------+
| liquid stain | 0.551 | 0.892  | 0.599  | 0.554 | 0.5   | 0.637 |
| congee stain | nan   | nan    | nan    | nan   | nan   | nan   |
| milk stain   | nan   | nan    | nan    | nan   | nan   | nan   |
| skein        | 0.436 | 0.692  | 0.491  | 0.0   | 0.321 | 0.602 |
| solid stain  | nan   | nan    | nan    | nan   | nan   | nan   |
+--------------+-------+--------+--------+-------+-------+-------+

```

---

