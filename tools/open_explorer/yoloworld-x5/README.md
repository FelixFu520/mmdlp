# yoloworld导出
- https://horizonrobotics.feishu.cn/docx/B4QUdAv0voKpdGxJcIucaKAynIh

```
PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/infer_float.py \
    --config /home/users/fa.fu/work/github/YOLO-World-x5/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    --img_path /home/users/fa.fu/work/work_dirs/yoloworld-x5/demo_images/meeting_room.jpg \
    --out_dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/result/ \
    --load_from /home/users/fa.fu/work/work_dirs/yoloworld-x5/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth \
    --work_dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/result/



```