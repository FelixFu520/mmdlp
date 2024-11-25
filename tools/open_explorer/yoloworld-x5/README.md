# yoloworld导出
- https://horizonrobotics.feishu.cn/docx/B4QUdAv0voKpdGxJcIucaKAynIh

```
# mm环境
PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/infer_float.py \
    --config /home/users/fa.fu/work/github/YOLO-World-x5/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    --img_path /home/users/fa.fu/work/work_dirs/yoloworld-x5/demo_images/meeting_room.jpg \
    --out_dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/ \
    --load_from /home/users/fa.fu/work/work_dirs/yoloworld-x5/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth \
    --work_dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/result/

PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/gen_class_json.py \
    --text "person,head,hand,arm,body,leg,foot,whiteboard,keyboard,mouse,laptop,marker pen,cup,bottle,eraser,microphone,mobile phone" \
    --output_file /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts.json


# 1. Prepare custom text embeddings 
PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/github/YOLO-World-x5/tools/generate_text_prompts.py \
    --model /home/users/fa.fu/work/work_dirs/yoloworld-x5/clip_vit_base_patch32 \
    --text /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts.json \
    --out /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts_17_texts_feats.npy

# 2. Reparameterizing
PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/github/YOLO-World-x5/tools/reparameterize_yoloworld.py \
    --model /home/users/fa.fu/work/work_dirs/yoloworld-x5/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth \
    --out-dir  /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/ \
    --text-embed /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts_17_texts_feats.npy \
    --conv-neck

# 3. Export onnx 
PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/github/YOLO-World-x5/deploy/export_onnx.py \
  /home/users/fa.fu/work/github/YOLO-World-x5/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival_custom-17.py \
  /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth \
  --work-dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/output \
  --img-size 640 640 \
  --batch 1 \
  --device cpu \
  --opset 11 \
  --without-nms

PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/github/YOLO-World-x5/deploy/onnx_demo.py \
    /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.onnx \
    /home/users/fa.fu/work/work_dirs/yoloworld-x5/demo_images/meeting_room.jpg \
    /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts.json \
    --onnx-nms \
    --device cpu \
    --output-dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer

hb_mapper makertbin -c /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/20241120/con_config_v1.yaml --model-type onnx

PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/infer_quant.py \
    --onnx_path /home/users/fa.fu/work/work_dirs/yoloworld-x5/20241120/output-v1/yolo_world_v2_s_int16_nv12_quantized_model.onnx \
    --input_image /home/users/fa.fu/work/work_dirs/yoloworld-x5/demo_images/meeting_room.jpg \
    --score_thr 0.21 \
    --output_image quant_meeting_root.jpg \
    --output_dir /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer \
    --class_names /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/custom_texts.json


scp /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/input_data.bin root@10.112.10.82:/root
scp /home/users/fa.fu/work/work_dirs/yoloworld-x5/20241120/output-v1/yolo_world_v2_s_int16_nv12.bin root@10.112.10.82:/root
hrt_model_exec infer --model_file yolo_world_v2_s_int16_nv12.bin --input_file input_data.bin --enable_dump --dump_format bin
scp root@10.112.10.82:/root/model_infer_output_1_boxes.bin /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/
scp root@10.112.10.82:/root/model_infer_output_0_scores.bin /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/

PYTHONPATH=/home/users/fa.fu/work/github/YOLO-World-x5 python /home/users/fa.fu/work/mmdlp/tools/open_explorer/yoloworld-x5/diff_board_host.py \
    --host_score /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/scores.npy \
    --host_bbox /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/bboxes.npy \
    --board_score /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/model_infer_output_0_scores.bin \
    --board_bbox /home/users/fa.fu/work/work_dirs/yoloworld-x5/output/onnx_infer/model_infer_output_1_boxes.bin

```