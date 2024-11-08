# 从预训练权重开始, 剪枝模型
```
# 单卡(训练+剪枝), 剪枝发生在after_train
python /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py /home/users/fa.fu/work/mmdlp/configs/mine/yolov5_pruning/yolov5-s-baseline-relu-train_pruning.py

# 多卡(训练+剪枝), 剪枝发生在after_train
PYTHONPATH=/home/users/fa.fu/work/mmdlp python -m torch.distributed.launch --nproc_per_node=8 --node_rank=0 --nnodes=1 --master_addr=127.0.0.1 --master_port=29501 /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py  /home/users/fa.fu/work/mmdlp/configs/mine/yolov5_pruning/yolov5-s-baseline-relu-train_pruning.py --work-dir /home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-train_pruning_multi --launcher pytorch

# 单卡(微调)
PYTHONPATH=/home/users/fa.fu/work/mmdlp python /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py /home/users/fa.fu/work/mmdlp/configs/mine/yolov5_pruning/yolov5-s-baseline-relu-finetune.py --work-dir /home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-finetune

# 多卡(微调)

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=/home/users/fa.fu/work/mmdlp python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --nnodes=1 --master_addr=127.0.0.1 --master_port=29501 /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py  /home/users/fa.fu/work/mmdlp/configs/mine/yolov5_pruning/yolov5-s-baseline-relu-finetune.py --work-dir /home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-finetune_multi --launcher pytorch

```