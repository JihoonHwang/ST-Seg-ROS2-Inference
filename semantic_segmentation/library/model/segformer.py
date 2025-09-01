# MMSegmentation 1.2.2 Compatible Configuration
# Based on MMEngine architecture

# Model settings
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth')),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Dataset settings
dataset_type = 'Dev_Large'
data_root = 'data/rugd_2x/'
crop_size = (512, 512)

# Data pipeline - Updated for MMSegmentation 1.x
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')  # Replaces DefaultFormatBundle + Collect
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 540), keep_ratio=True),
    dict(type='LoadAnnotations'),  # for test-time evaluation
    dict(type='PackSegInputs')
]

# DataLoader settings
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img',
            seg_map_path='gt'),
        ann_file='train_rugdrellisgoose.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img',
            seg_map_path='gt'),
        ann_file='test_rugdrellisgoose_v2.txt',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # RGR-C configuration
        data_root='data/rgr-c/',
        data_prefix=dict(
            img_path='RGR_frost-5',
            seg_map_path='../rugd_2x/gt'),
        ann_file='../rugd_2x/test_rugdrellisgoose_v2.txt',
        pipeline=test_pipeline))

# Evaluation settings
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Optimizer and scheduler settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-06,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]

# Training settings
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=16000)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings
default_scope = 'mmseg'

# Environment settings
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Logging settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=False)

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', by_epoch=False)  # Optional
    ])

# Checkpoint settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Custom imports - 커스텀 데이터셋을 사용하는 경우 실제 경로로 수정 필요
# custom_imports = dict(imports=['path.to.your.custom_dataset'], allow_failed_imports=False)

# Resume and load settings
resume = False  # Changed from resume_from
load_from = None

# Work directory
work_dir = './dev_cl_work_dirs/final_large/mitb0_512x512_rugdrellisgoose_large'

# Set random seed for reproducibility
randomness = dict(seed=None)