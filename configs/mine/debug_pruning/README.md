# 剪枝试验
## 方案1: 多次剪枝(单卡)
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/users/fa.fu/work/mmdlp python /home/users/fa.fu/work/mmdlp/mmdlp/tools/classify/train.py /home/users/fa.fu/work/mmdlp/configs/mine/debug_pruning/resnet34_8xb16_cifar10_trainfinetune_singlegpu_v2.py
```
## 方案2: 多次剪枝(多卡)
多卡运行失败
```
PYTHONPATH=/home/users/fa.fu/work/mmdlp python /usr/local/lib/python3.8/site-packages/torch/distributed/launch.py --nproc_per_node=8 --node_rank=0 /home/users/fa.fu/work/mmdlp/configs/mine/debug_pruning/resnet34_8xb16_cifar10_trainfinetune_multigpu_v1.py
```
## 方案3: 单次剪枝
```

```