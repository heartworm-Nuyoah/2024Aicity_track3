'''
@Project :Video-Swin-Transformer 
@File    :swin_tiny_meva_pp.py
@IDE     :PyCharm 
@Author  :ycxiao@bupt.edu.cn
@Date    :2022/9/16 12:31 
@brief   :A file to do some processes.
'''
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=491,#TODO
        # loss_cls=dict(type='BCELossWithLogits'),
        spatial_type='avg',
        multi_class=True,
        dropout_ratio=0.5),
    # test_cfg=dict(average_clips='prob', max_testing_views=4))
    test_cfg=dict(average_clips='sigmod'))
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
resume_from = '/home/zhp/GITHUB/mmaction2-master/work_dirs/1120/epoch_3.pth'
workflow = [('train', 1)]
num_classes=491#
dataset_type = 'FadRawframeDataset'

# data_root_train = '/home/xyc/data/OpenFAD/cap_classification_clip/videos'
# 这三个路径是一样的

data_root_train = '/data1/zhp/OpenFAD/cap_classification_clip/videos'
data_root_val = '/data1/zhp/OpenFAD/cap_classification_clip/videos'
data_root_test = '/data1/zhp/OpenFAD/cap_classification_clip/videos'

# ann_file_train = '/home/xyc/data/OpenFAD/cap_classification_clip/train_clip.json'#TODO:
# ann_file_val = '/home/xyc/data/OpenFAD/cap_classification_clip/val_clip.json'
# ann_file_test = '/home/xyc/data/OpenFAD/cap_classification_clip/val_clip.json'

ann_file_train = '/data1/zhp/OpenFAD/cap_classification_clip/train_clip.json'#TODO:
ann_file_val = '/data1/zhp/OpenFAD/cap_classification_clip/val_clip.json'
ann_file_test = '/data1/zhp/OpenFAD/cap_classification_clip/val_clip.json'

# dict_util = '/data/OpenFAD/cap_classification_clip/train_clip_classes_index.json' # Classes id index
dict_util = '/data1/zhp/OpenFAD/cap_classification_clip/train_clip_classes_index.json' # Classes id index

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='FadSampleFrames', clip_len=8, frame_interval=4, num_clips=1),
    dict(type='RawVideoDecode', pad_square = True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='FadSampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawVideoDecode', pad_square = True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='FadSampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawVideoDecode', pad_square = True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=16,#16
    workers_per_gpu=2,#2
    # val_dataloader=dict(videos_per_gpu=256, workers_per_gpu=16),
    # test_dataloader=dict(videos_per_gpu=256, workers_per_gpu=16),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root_train,
        pipeline=train_pipeline,
        dict_util=dict_util,
        multi_class=True,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        dict_util = dict_util,
        multi_class=True,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,#ann_file_val,
        data_prefix=data_root_test,
        pipeline=test_pipeline,
        dict_util = dict_util,
        multi_class=True,
        num_classes=num_classes))
evaluation = dict(
    interval=4, metrics=['top_k_accuracy', 'mean_average_precision'])
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=3)
total_epochs = 10 #30
work_dir = None
find_unused_parameters = False
fp16 = None

optimizer_config = dict(grad_clip=None)

gpu_ids = range(0, 5)
# gpu_ids = [2,3,4,5,6]
omnisource = False
module_hooks = []
