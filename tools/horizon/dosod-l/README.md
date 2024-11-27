# 对齐pth, float onnx, qunat onnx的结果
详细见 https://horizonrobotics.feishu.cn/docx/MW7jdGbs5o0oUuxX8h7cWKruntb

## 1 对齐pth与float onnx
```
# 跑出pth的mAP
cd /home/fa.fu/work/horizon/YOLO-World-dosod
PYTHONPATH=/home/fa.fu/work/horizon/YOLO-World-dosod:/home/fa.fu/work/horizon/YOLO-World-dosod/mmdetection python3 /home/fa.fu/work/horizon/YOLO-World-dosod/tools/test.py \
    /home/fa.fu/work/horizon/YOLO-World-dosod/configs/kexuejia_1113/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p_science.py  \
    /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_motionblur_20241113_1024x1024_672x896.pth \
    --work-dir /home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p \
    --out /home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/pth5.pkl

# 对比输入
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/preprocess.py \
    --height 672 \
    --width 896 \
    --images_dir /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/real_resize_jpg_data_20241103

# 推理float onnx
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_image_onnx.py \
    --height 672 \
    --width 896 \
    --image_path /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg \
    --result_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896.onnx

# 推理pth
PYTHONPATH=/home/fa.fu/work/horizon/YOLO-World-dosod python3 /home/fa.fu/work/horizon/YOLO-World-dosod/demo/image_demo.py \
    /home/fa.fu/work/mmdlp/configs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p.py \
    /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_motionblur_20241113_1024x1024_672x896.pth \
    --output-dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result/pth

# 验证mAP
PYTHONPATH=/home/fa.fu/work/mmdlp python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_images_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/real_resize_jpg_data_20241103 \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896.onnx \
    --save_dir_float /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result/float_npy \
    --show_dir eval_result_show_float_data20241103 \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/eval_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result/float_npy \
    --height 672 \
    --width 896

# 验证召回率 pth
PYTHONPATH=/home/fa.fu/work/horizon/YOLO-World-dosod:/home/fa.fu/work/horizon/YOLO-World-dosod/mmdetection python3 /home/fa.fu/work/horizon/YOLO-World-dosod/tools/test.py \
    /home/fa.fu/work/horizon/YOLO-World-dosod/configs/kexuejia_1113/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p_science.py  \
    /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_motionblur_20241113_1024x1024_672x896.pth \
    --work-dir /home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p \
    --out /home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/pth5.pkl
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/pr_curve_kexuejia_test.py

# 验证召回率 onnx
PYTHONPATH=/home/fa.fu/work/mmdlp python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_images_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset/real_resize_jpg_data \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896.onnx \
    --save_dir_float /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result/float_npy_reinjection \
    --show_dir eval_result_show_float_data_reinjection \
    --height 672 \
    --width 896
PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/eval_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset \
    --ann_file real_resize_coco_jpg.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result/float_npy_reinjection \
    --height 672 \
    --width 896
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/pr_curve_kexuejia_test.py
```

## 2. 对齐float onnx与int16
```
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/gen_calibration_data.py    \
    --data_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_75\
    --save_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/calibration_data/calibration_images_75_featuremap_672×896 \
    --height 672 \
    --width 896

hb_mapper makertbin -c /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/20241126/con_DOSOD_L_v1.yaml --model-type onnx

python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_image_onnx.py \
    --height 672 \
    --width 896 \
    --image_path /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images/0892.jpg \
    --result_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/demo_images_result \
    --onnx_float_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_calibrated_model.onnx

PYTHONPATH=/home/fa.fu/work/mmdlp python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_images_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/real_resize_jpg_data_20241103 \
    --onnx_calib_featuremap_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_calibrated_model.onnx \
    --save_dir_calib_featuremap /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/v1_data20241103_calib_npy \
    --show_dir v1_data20241103_calib_show \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/eval_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/ \
    --ann_file real_resize_coco_jpg_20241103.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/v1_data20241103_calib_npy \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/infer_images_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset/real_resize_jpg_data \
    --onnx_calib_featuremap_path /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/output_v1/dosod-l_epoch_40_kxj_rep-without-nms_motionblur_20241113_1024x1024_672x896_calibrated_model.onnx \
    --save_dir_calib_featuremap /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/v1_reinjection_calib_npy \
    --show_dir v1_reinjection_calib_show \
    --height 672 \
    --width 896

PYTHONPATH=/home/fa.fu/work/mmdlp/ python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/eval_onnx.py \
    --data_dir /horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset/ \
    --ann_file real_resize_coco_jpg.json \
    --pred_npy_dir /home/fa.fu/work/work_dirs/horizon/dosod-l/20241126/v1_reinjection_calib_npy \
    --height 672 \
    --width 896
mv test.pkl /home/fa.fu/work/work_dirs/horizon/dosod-l/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/v1_calib.pkl
python3 /home/fa.fu/work/mmdlp/tools/horizon/dosod-l/pr_curve_kexuejia_test.py
```