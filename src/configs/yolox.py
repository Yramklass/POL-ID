# Model Configuration
_base_ = '/home/rmkyas002/pol_id/detection/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'

# Dataset
data_root = '/scratch/rmkyas002/mmdetection_data/'
class_name = ('pollen',)
metainfo = dict(classes=class_name)

# Image Size
img_scale = (640, 640)

# 3. Pipeline Definitions (Updated)
train_pipeline = [
    # The 'LoadImageFromFile' transform is now required before Mosaic/MixUp
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Data Loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
                # (No MixUp or Mosaic here!)
            ]
        ),
        pipeline=[  # mix transforms go here!
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ]
    )
)


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline
    )
)


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline
    )
)

# Model Head
model = dict(
    bbox_head=dict(num_classes=1),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# Training Schedule
max_epochs = 100
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - 10, 1)])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[80, 90],
        gamma=0.1)
]

# Runtime
default_scope = 'mmdet'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                 dict(type='TensorboardVisBackend')],
    name='visualizer')

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    format_only=False)