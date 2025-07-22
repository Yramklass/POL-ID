# Model Configuration
_base_ = 'yolox/yolox_s_8xb8-300e_coco.py'

# 1. Dataset Settings
data_root = '/scratch/rmkyas002/mmdetection_data/'
class_name = ('pollen',)
num_classes = len(class_name)
metainfo = dict(classes=class_name)
img_scale = (640, 640)

# 2. Model Head Configuration
model = dict(
    bbox_head=dict(num_classes=num_classes),
    train_cfg=dict(assigner=dict(center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# 3. Pipeline Definitions
train_pipeline = [
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 4. Dataloader Definitions
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 5. Evaluator Definitions
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox')
test_evaluator = val_evaluator

# 6. Training Schedule and Runtime
max_epochs = 100
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - 10, 1)]) # Close mosaic aug near the end

# Use auto_scale_lr in the sbatch script instead of hardcoding LR
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4), # Base LR
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[80, 90],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto', max_keep_ckpts=3))