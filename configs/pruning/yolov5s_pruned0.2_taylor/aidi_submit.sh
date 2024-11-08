pip install ftfy==6.2.3 regex==2024.9.11 torch-pruning==1.4.3 thop==0.1.1.post2209072238
PYTHONPATH=/running_package/code_package/ python -m torch.distributed.launch \
                                   --nnodes=1 --node_rank=0 \
                                   --master_addr=127.0.0.1 --nproc_per_node=8 \
                                   --master_port=10011 \
                                   /running_package/code_package/mmdlp/tools/detect/train.py \
                                   /running_package/code_package/configs/pruning/yolov5s_pruned0.2_taylor/aidi_submit_yolov5-s-baseline-relu-finetune-taylor.py \
                                   --launcher pytorch \
                                   --work-dir /horizon-bucket/AIoT-data-bucket/fa.fu/work_dirs/aidi_submit_yolov5-s-baseline-relu-finetune-taylor
