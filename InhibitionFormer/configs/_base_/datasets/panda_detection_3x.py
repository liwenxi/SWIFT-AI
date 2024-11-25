# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(256 * 14, 256 * 7)),
    # dict(type='Resize', img_scale=(512 * 4, 512 * 2), keep_ratio=True),
    dict(type='Resize', img_scale=(256 * 14, 256 * 7), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256 * 14, 256 * 7),
        # img_scale=(512 * 4, 512 * 2),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('person',)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/train_s4.json',
            img_prefix='/home/liwenxi/panda/raw/PANDA/patches/s4',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/val_s4.json',
        img_prefix='/home/liwenxi/panda/raw/PANDA/patches/s4',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/val_s4.json',
        img_prefix='/home/liwenxi/panda/raw/PANDA/patches/s4',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox')
