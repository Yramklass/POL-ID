# Inherit from the base YOLOX config
_base_ = '/home/rmkyas002/pol_id/detection/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'

# DATASET CONFIGURATION 
data_root = '/scratch/rmkyas002/mmdetection_data/'
class_name = ('pollen',)
num_classes = len(class_name)
img_scale = (640, 640)

# Pipeline for training 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.1, 2), border=(-img_scale[0]//2, -img_scale[1]//2)),
    dict(type='MixUp', img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

# Pipeline for validation and testing
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataset configuration
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=class_name),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=dict(classes=class_name),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=class_name),
        test_mode=True,
        pipeline=test_pipeline))

# MODEL CONFIGURATION 
model = dict(
    bbox_head=dict(num_classes=num_classes),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# TRAINING CONFIGURATION 
max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[80, 90], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        rule='greater',
        max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

# RUNTIME CONFIGURATION 
work_dir = '/scratch/rmkyas002/detection_outputs/pollen_yolox_run1'
randomness = dict(seed=42, deterministic=False)

# EVALUATION CONFIGURATION 
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