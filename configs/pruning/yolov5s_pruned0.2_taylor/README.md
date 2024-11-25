## run
从预训练权重开始, 剪枝模型. 训练结果存在/home/users/fa.fu/work/work_dirs/yolov5-s
```
# 单卡(训练+剪枝), 剪枝发生在after_train
PYTHONPATH=/home/users/fa.fu/work/mmdlp python /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py \
       /home/users/fa.fu/work/mmdlp/configs/pruning/yolov5s_pruned0.2_taylor/yolov5-s-baseline-relu-resume_for_pruning-taylor.py

# 多卡(训练+剪枝), 剪枝发生在after_train
PYTHONPATH=/home/users/fa.fu/work/mmdlp python -m torch.distributed.launch \
       --nproc_per_node=8 --node_rank=0 --nnodes=1 --master_addr=127.0.0.1 --master_port=29501 \
       /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py  \
       /home/users/fa.fu/work/mmdlp/configs/pruning/yolov5s_pruned0.2_taylor/yolov5-s-baseline-relu-resume_for_pruning-taylor.py \
       --work-dir /home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-resume_for_pruning-taylor_multi \
       --launcher pytorch

# 单卡(微调)
PYTHONPATH=/home/users/fa.fu/work/mmdlp python /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py \
       /home/users/fa.fu/work/mmdlp/configs/pruning/yolov5s_pruned0.2_taylor/yolov5-s-baseline-relu-finetune-taylor.py

# 多卡(微调)
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=/home/users/fa.fu/work/mmdlp python -m torch.distributed.launch \
       --nproc_per_node=4 --node_rank=0 --nnodes=1 --master_addr=127.0.0.1 --master_port=29501 \
       /home/users/fa.fu/work/mmdlp/mmdlp/tools/detect/train.py  \
       /home/users/fa.fu/work/mmdlp/configs/pruning/yolov5s_pruned0.2_taylor/yolov5-s-baseline-relu-finetune-taylor.py \
       --work-dir /home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-finetune-taylor_multi \
       --launcher pytorch

# 提交到aidi平台
pip3 install aidisdk==0.18.0 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple

aidi-inf-cli init -t eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIzNTk2MTI4MDUsIlRva2VuVHlwZSI6ImxkYXAiLCJVSUQiOjEyMzg0LCJIYXNoVUlEIjoxOTI2MzUwODMwLCJVc2VyTmFtZSI6ImZhLmZ1IiwiRW1haWwiOiIiLCJUZW5hbnQiOiJyZWd1bGFyLWVuZ2luZWVyIiwiVGVuYW50SUQiOjEsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MSwiVmVyc2lvbiI6InYyIn0.zNnuMmdQ4UdqkubWiAjVDfA-PabL5I7y880udxZ9pYJsbj7JNG1Vol9GN6kNlCTHVWkiBW3InO_m9PMR9PwfBkvWRqkp5wuNXYmL4IAl7YPIfISF1w_KkF4tJZ316g2y4K5lJz0kmZH7JoWY87AAJrDoHAVC5ogdnDYCSiR-NP56PVlnmxscl8kgLx37fMHPStdGwbWe2Uv9zNNaC6ehudymzxZO64sSL2vYDS-vdb7WCNKZOwQvQijqn3LI89XHrsSC-kw1qF2u5GNQJV9htXh7_u0ejl8B2pKdQ_idUyYIcRMpSZhBghV8zAWW6l_OAY7QUhyf7YFGCF9MAis_rA

python3 aidi_submit.py
```