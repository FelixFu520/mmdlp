cd /running_package/code_package 
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=8 --master_port=10010 train.py yolov5-1.5n-baseline-relu.py --launcher pytorch --work-dir /horizon-bucket/AIoT-data-bucket/fa.fu/work_dir/yolov5-1.5n
