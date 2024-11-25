# ----------------- model 变量
deepen_factor = 0.33    # The scaling factor that controls the depth of the network structure
widen_factor = 0.5      # The scaling factor that controls the width of the network structure
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)    # Normalization config
num_classes = 17    # Number of classes for classification
# Basic size of multi-scale prior box
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
strides = [8, 16, 32]   # Strides of multi-scale prior box
loss_cls_weight = 0.5   # Loss weight of classification loss
loss_bbox_weight = 0.05 # Loss weight of bounding box loss
num_det_layers = 3      # The number of model output scales
loss_obj_weight = 1.0   # loss weight of objectness loss
prior_match_thr = 4.    # Priori box matching threshold
obj_level_weights = [4., 1., 0.4]   # The obj loss weights of the three output layers


# ----------------- dataset变量
data_root = '/home/users/fa.fu/work/data/seed_dataset/'  # Root path of data
train_ann_file = 'seed_coco_80.json'    # Path of train annotation file
train_data_prefix = ''  # Prefix of train image path
val_ann_file = 'seed_coco_20.json'  # Path of val annotation file
val_data_prefix = ''  # Prefix of val image path
train_batch_size_per_gpu = 16   # Batch size of a single GPU during training
train_num_workers = 8   # Worker to pre-fetch data for each single GPU during training
persistent_workers = True   # persistent_workers must be False if num_workers is 0
dataset_type = 'YOLOv5CocoDataset'  # Dataset type, this will be used to define the dataset
img_scale = (640, 640)  # image scale, width, height
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
val_batch_size_per_gpu = 1  # Batch size of a single GPU during validation
val_num_workers = 2 # Worker to pre-fetch data for each single GPU during validation


# --------------- train/val/test&optim&scheduler&evaluator&runtime 变量
backend_args = None
base_lr = 0.01  # Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
max_epochs = 100  # Maximum training epochs
weight_decay = 0.0005
lr_factor = 0.01  # Learning rate scaling factor
save_checkpoint_intervals = 1  # Save model checkpoint and validation intervals
default_scope = 'mmyolo'
log_level = 'INFO'
load_from = None
resume = None
# runner settings
runner_type = 'mmdlp.PruningRunner'

# ****************************************** model settings ******************************************
model = dict(
    path='/home/users/fa.fu/work/work_dirs/yolov5-s-baseline-relu-resume_for_pruning-taylor/pruned_model.pth',
    special=True
    )

# ***************************************** dataset settings *******************************
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('yellowRust',
                               'impurity',
                               'trigone',
                               'smallGrain',
                               'smallWhite',
                               'bigGrain',
                               'rust',
                               'bigWhite',
                               'badSeed',
                               'macula',
                               'largeMacula',
                               'break',
                               'twins',
                               'crack',
                               'worm',
                               'whiteHead',
                               'moldy')),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('yellowRust',
                               'impurity',
                               'trigone',
                               'smallGrain',
                               'smallWhite',
                               'bigGrain',
                               'rust',
                               'bigWhite',
                               'badSeed',
                               'macula',
                               'largeMacula',
                               'break',
                               'twins',
                               'crack',
                               'worm',
                               'whiteHead',
                               'moldy')),
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader


# ***************************************** train&optimizer&scheduler&evaluate&log&visualizer settings *******************************
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    classwise=True,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        save_best=None,
        rule=None,
        max_keep_ckpts=-1),
    )
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)


# ***************************************** runtime settings ***********************************************
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
randomness = dict(
    seed = 1024,
    deterministic=True,
    diff_rank_seed=False
)
