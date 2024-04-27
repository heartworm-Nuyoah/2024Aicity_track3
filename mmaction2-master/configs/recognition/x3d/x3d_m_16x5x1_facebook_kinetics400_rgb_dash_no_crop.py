_base_ = ['/disk2/js_data/AICITY/2023code/mmaction2-master/configs/_base_/models/x3d.py',
        '/disk2/js_data/AICITY/2023code/mmaction2-master/configs/_base_/default_runtime.py',
]
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        # in_channels=432,
        num_classes=17,
        # spatial_type='avg',
        # dropout_ratio=0.5,
        fc1_bias=False),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
# model=dict(cls_head=dict(num_classes=17))
# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/disk2/js_data/AICITY/datasets/2023/AI-City-Challenge-2023/A1_frame'
data_root_val = '/disk2/js_data/AICITY/datasets/2023/AI-City-Challenge-2023/A1_frame'


ann_file_train = '/disk2/js_data/AICITY/datasets/2023/AI-City-Challenge-2023/annotation/dash_01_anno_train.txt'
ann_file_val = '/disk2/js_data/AICITY/datasets/2023/AI-City-Challenge-2023/annotation/dash_01_anno_val.txt'
ann_file_test = '/raid/aicity/data/annotations/k_fold/dash_24026_anno_ext6_val.txt'
img_norm_cfg = dict(
    mean=[86.95, 86.95, 86.95], std=[72.59, 72.59, 72.59], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode', pad = False, crop = False, reverse = 0.0),
    # dict(type='Resize', scale=(-1, 512)),
    # dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(480, 480), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1,test_mode=True),
    dict(type='RawFrameDecode', pad = False, crop = False, reverse = 0.0),
    dict(type='Resize', scale=(480, 480), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='RawframeDataset'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', pad = False, crop = False, reverse = 0.0),
    dict(type='Resize', scale=(480, 480), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=2,
        workers_per_gpu=4
    ),
    test_dataloader=dict(
        videos_per_gpu=2,
        workers_per_gpu=4
    ),
    train=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        filename_tmpl='frame{:06}.jpg',
        with_offset=True,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy','mean_class_accuracy'])
checkpoint_config = dict(interval=1)    
load_from = '/disk2/js_data/AICITY/2023code/mmaction2-master/checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
dist_params = dict(backend='nccl')
# optimizer = dict(type='AdamW')
optimizer = dict(type='AdamW', lr=1e-3*1.5, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2
)
total_epochs = 20
# checkpoint_config = dict(  # 模型权重文件钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
#     interval=1)   # 模型权重文件保存间隔