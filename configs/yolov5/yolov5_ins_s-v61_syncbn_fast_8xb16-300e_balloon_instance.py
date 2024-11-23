_base_ = '/usr/local/lib/python3.8/dist-packages/mmyolo/.mim/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'


data_root = '/root/data/datasets/balloon/data/balloon/'
# 训练集标注路径
train_ann_file = 'train.json'
train_data_prefix = 'train/'  # 训练集图片路径
# 测试集标注路径
val_ann_file = 'val.json'
val_data_prefix = 'val/'  # 验证集图片路径
metainfo = {
    'classes': ('balloon', ), # 数据集类别名称
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1
# 批处理大小batch size设置为 4
train_batch_size_per_gpu = 4
# dataloader 加载进程数
train_num_workers = 2
log_interval = 1
#####################
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
default_hooks = dict(logger=dict(interval=log_interval))
#####################

model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
