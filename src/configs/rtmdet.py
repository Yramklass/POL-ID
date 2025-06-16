# Inherit from the base RTMDet-tiny config
_base_ = '/home/rmkyas002/pol_id/detection/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'


# DATASET & EVALUATION CONFIGURATION 
data_root = '/scratch/rmkyas002/mmdetection_data/' 


class_name = ('pollen',)
# The number of classes
num_classes = len(class_name) 

train_batch_size_per_gpu = 16 
train_num_workers = 1 # Adjust based on HPC core availability
# Data augmentation
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.5, 1.5)),
    dict(type='MixUp', img_scale=(640, 640), ratio_range=(0.8, 1.6)),
    dict(type='RandomFlip', prob=0.5),  # Default RTMDet augmentation
    dict(type='Resize', scale=(640, 640), keep_ratio=True),  # Resize to input size
    dict(type='PhotoMetricDistortion'),  # Color jitter
    dict(type='PackDetInputs')  # Required for MMDetection
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=class_name),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=dict(classes=class_name)
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=class_name)
    )
)


val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    format_only=False
)


# MODEL CONFIGURATION 
model = dict(bbox_head=dict(num_classes=num_classes))

# TRAINING SCHEDULE & RUNTIME CONFIGURATION 
max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=5)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=None
)


val_cfg = dict()
test_cfg = dict()


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto', 
        rule='greater',
        max_keep_ckpts=3
    )
)

# Directory where logs and models will be saved
work_dir = '/scratch/rmkyas002/detection_outputs/pollen_rtmdet_run1'

# Seeded for reproducibility
randomness = dict(seed=42)