pip install ftfy==6.2.3 regex==2024.9.11 torch-pruning==1.4.3 thop==0.1.1.post2209072238
PYTHONPATH=/running_package/code_package/ python -m torch.distributed.launch \
                                   --nnodes=1 --node_rank=0 \
                                   --master_addr=127.0.0.1 --nproc_per_node=8 \
                                   --master_port=10011 \
                                   /running_package/code_package/mmdlp/tools/detect/train.py \
                                   /running_package/code_package/configs/mine/yolov5_pruning_aidi/yolov5-s-baseline-relu-finetune-gn.py \
                                   --launcher pytorch \
                                   --work-dir /horizon-bucket/AIoT-data-bucket/fa.fu/work_dir/yolov5-s-pruning-finetune-300epoch-gn
