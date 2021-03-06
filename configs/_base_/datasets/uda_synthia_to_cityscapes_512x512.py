# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPanopticImage'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='LoadAnnotationsPanopticData'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_pan_data']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPanopticImage'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='LoadAnnotationsPanopticData'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_pan_data']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaDataset',
            data_root='/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/synthia/',
            img_dir='RGB',
            ann_dir='panoptic-labels-crowdth-0-for-daformer',
            aux_dir='synthia_panoptic',
            dict_dir='synthia_panoptic.json',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root='/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine_panoptic',
            aux_dir='cityscapes_panoptic_train_trainId',
            dict_dir='cityscapes_panoptic_train_trainId.json',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine_panoptic/cityscapes_panoptic_val',
        pipeline=test_pipeline))
