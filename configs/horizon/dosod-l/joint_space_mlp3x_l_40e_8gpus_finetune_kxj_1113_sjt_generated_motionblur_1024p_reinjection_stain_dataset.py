_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.9
albu_train_transforms = []
backend_args = None
base_lr = 0.0002
batch_shapes_cfg = None
close_mosaic_epochs = 10
coco_train_dataset = dict(
    _delete_=True,
    class_text_path='data/texts/kxj_class_texts_1021.json',
    dataset=dict(
        ann_file='real_virtual_resize_coco_jpg_20241113.json',
        data_prefix=dict(img=''),
        data_root=
        '/horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/train_stain_dataset',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'liquid stain',
                'congee stain',
                'milk stain',
                'skein',
                'solid stain',
            )),
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                1024,
                1024,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(prob=0.3, type='YOLOv5CopyPaste'),
        dict(
            border=(
                -512,
                -512,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    img_scale=(
                        1024,
                        1024,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    type='MultiModalMosaic'),
                dict(prob=0.3, type='YOLOv5CopyPaste'),
                dict(
                    border=(
                        -512,
                        -512,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_aspect_ratio=100,
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    scaling_ratio_range=(
                        0.09999999999999998,
                        1.9,
                    ),
                    type='YOLOv5RandomAffine'),
            ],
            prob=0.15,
            type='YOLOv5MultiModalMixUp'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=5,
            num_neg_samples=(
                5,
                5,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
coco_val_dataset = dict(
    _delete_=True,
    class_text_path='data/texts/kxj_class_texts_1021.json',
    dataset=dict(
        ann_file='real_resize_coco_jpg.json',
        data_prefix=dict(img=''),
        data_root=
        '/horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'liquid stain',
                'congee stain',
                'milk stain',
                'skein',
                'solid stain',
            )),
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            896,
            672,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                896,
                672,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
copypaste_prob = 0.3
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=30,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                1024,
                1024,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    1024,
                    1024,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(
                        max_factor=(
                            1.02,
                            1.02,
                        ),
                        p=0.3,
                        step_factor=0.01,
                        type='ZoomBlur'),
                    dict(blur_limit=(
                        7,
                        7,
                    ), p=0.3, type='MotionBlur'),
                ],
                type='mmdet.Albu'),
            dict(
                max_num_samples=5,
                num_neg_samples=(
                    5,
                    5,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'yolo_world',
    ])
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1.0
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=-1,
        rule=None,
        save_best=None,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.1,
        max_epochs=40,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook',
        warmup_epochs=5,
        warmup_mim_iter=0),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    1024,
    1024,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
joint_space_dims = 512
last_stage_out_channels = 512
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'pytorch'
load_from = '/horizon-bucket/d-robotics-bucket/yonghao01.he/workspace/train_jobs/joint_space_mlp3x_l_100e_1x8gpus_obj365v1_goldg_train_lvis_minival/2024_08_22_15_42_10/epoch_100.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 40
max_keep_ckpts = 2
mixup_prob = 0.15
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=1.0,
        last_stage_out_channels=512,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=1.0),
    backbone_text=dict(
        frozen_modules=[
            'all',
        ],
        model_name=
        '/horizon-bucket/d-robotics-bucket/yonghao01.he/pretrain_models/clip-vit-base-patch32',
        type='HuggingCLIPLanguageBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            joint_space_dims=512,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=5,
            num_text_joint_learning_layers=3,
            reg_max=16,
            text_embed_dims=512,
            type='JointSpaceYOLOv8dHeadModule',
            widen_factor=1.0),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='JointSpaceYOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=1.0,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            512,
        ],
        type='YOLOv8PAFPN',
        widen_factor=1.0),
    num_test_classes=5,
    num_train_classes=5,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(class_agnostic=True, iou_threshold=0.4, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=5,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='JointSpaceDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(class_agnostic=True, iou_threshold=0.4, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(prob=0.3, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    256,
]
neck_num_heads = [
    4,
    8,
    8,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 5
num_det_layers = 3
num_text_joint_learning_layers = 3
num_training_classes = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOWv5OptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        batch_size_per_gpu=12, lr=0.0002, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        })),
    type='AmpOptimWrapper')
param_scheduler = None
persistent_workers = False
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
randomness = dict(seed=3407)
resume = False
save_epoch_intervals = 5
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/kxj_class_texts_1021.json',
        dataset=dict(
            ann_file='real_resize_coco_jpg.json',
            data_prefix=dict(img=''),
            data_root=
            '/horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'liquid stain',
                    'congee stain',
                    'milk stain',
                    'skein',
                    'solid stain',
                )),
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                896,
                672,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    896,
                    672,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/horizon-bucket/d-robotics-bucket/fa.fu/dosod-l/reinjection_stain_dataset/real_resize_coco_jpg.json',
    classwise=True,
    metric='bbox',
    pr_plot=True,
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        896,
        672,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            896,
            672,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 512
text_model_name = '/horizon-bucket/d-robotics-bucket/yonghao01.he/pretrain_models/clip-vit-base-patch32'
text_transform = [
    dict(
        max_num_samples=5,
        num_neg_samples=(
            5,
            5,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 12
train_cfg = dict(
    dynamic_intervals=[
        (
            490,
            1,
        ),
    ],
    max_epochs=40,
    type='EpochBasedTrainLoop',
    val_interval=5)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=12,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        class_text_path='data/texts/kxj_class_texts_1021.json',
        dataset=dict(
            ann_file='real_virtual_resize_coco_jpg_20241113.json',
            data_prefix=dict(img=''),
            data_root=
            '/horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/train_stain_dataset',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'liquid stain',
                    'congee stain',
                    'milk stain',
                    'skein',
                    'solid stain',
                )),
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    1024,
                    1024,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='MultiModalMosaic'),
            dict(prob=0.3, type='YOLOv5CopyPaste'),
            dict(
                border=(
                    -512,
                    -512,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        img_scale=(
                            1024,
                            1024,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                        ],
                        type='MultiModalMosaic'),
                    dict(prob=0.3, type='YOLOv5CopyPaste'),
                    dict(
                        border=(
                            -512,
                            -512,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine'),
                ],
                prob=0.15,
                type='YOLOv5MultiModalMixUp'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=5,
                num_neg_samples=(
                    5,
                    5,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=8,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(prob=0.3, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    1024,
                    1024,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='MultiModalMosaic'),
            dict(prob=0.3, type='YOLOv5CopyPaste'),
            dict(
                border=(
                    -512,
                    -512,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
        ],
        prob=0.15,
        type='YOLOv5MultiModalMixUp'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=5,
        num_neg_samples=(
            5,
            5,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        1024,
        1024,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1024,
            1024,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(
                max_factor=(
                    1.02,
                    1.02,
                ),
                p=0.3,
                step_factor=0.01,
                type='ZoomBlur'),
            dict(blur_limit=(
                7,
                7,
            ), p=0.3, type='MotionBlur'),
        ],
        type='mmdet.Albu'),
    dict(
        max_num_samples=5,
        num_neg_samples=(
            5,
            5,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/kxj_class_texts_1021.json',
        dataset=dict(
            ann_file='real_resize_coco_jpg_20241103.json',
            data_prefix=dict(img=''),
            data_root=
            '/horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'liquid stain',
                    'congee stain',
                    'milk stain',
                    'skein',
                    'solid stain',
                )),
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                896,
                672,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    896,
                    672,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/test_stain_dataset/real_resize_coco_jpg_20241103.json',
    classwise=True,
    metric='bbox',
    pr_plot=True,
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_img_scale = (
    896,
    672,
)
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
widen_factor = 1.0
work_dir = '/horizon-bucket/d-robotics-bucket/shiyuan.chen/workspace/train_jobs/kexuejia/joint_space_mlp3x_l_40e_8gpus_finetune_kxj_1113_sjt_generated_motionblur_1024p/2024_11_14_01_20_05'
