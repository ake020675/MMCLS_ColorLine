"""
by zk
modified from mobilenet-v2_8xb32_imagenet.py
"""

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,  # num_classes=1000
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
    ))


# dataset settings
# albu_transform
albu_train_transforms = [

    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.5),
    # dict(type='RandomSunFlare', flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6,
    #      num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
    # dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=0, val_shift_limit=0,
    #      always_apply=False, p=0.5),  # 色调
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=7, p=1.0),
    #         dict(type='MedianBlur', blur_limit=7, p=1.0)
    #     ],
    #     p=0.2),
    # dict(type='ChannelShuffle', p=0.5),  # 色调

    dict(type='Rotate',
         limit=15,
         interpolation=1,
         border_mode=0,
         value=(0, 0, 0),
         p=0.5, ),  # 旋转

    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0, hue=0.2, always_apply=False, p=0.5),
    dict(type='MotionBlur', blur_limit=5, p=0.2),
    dict(type='GaussianBlur', blur_limit=(5, 5), p=0.2),
    dict(type='RandomScale', scale_limit=0.2, interpolation=1, p=0.5),  # 图像尺寸缩放0.2

    dict(type='PadIfNeeded', min_height=192, min_width=256, border_mode=0,
         value=(0, 0, 0), always_apply=False, p=1),  # 中心pad
]

albu_test_transforms = [
    dict(type='PadIfNeeded', min_height=192, min_width=256, border_mode=0,
         value=(0, 0, 0), always_apply=False, p=1),  # 中心pad
]

dataset_type = 'LaserLabel'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),

    # dict(type='Resize', size=(180, 240)),
    # dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0),  # 明亮度、对比度
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(
            type='Albu',
            transforms=albu_train_transforms,
            keymap={
                'img': 'image',
            },
            update_pad_shape=False,
            # skip_img_without_anno=True
        ),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(180, 240)),
    dict(
        type='Albu',
        transforms=albu_test_transforms,
        keymap={
            'img': 'image',
        },
        update_pad_shape=False,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400',
        ann_file='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400',
        ann_file='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400',
        ann_file='C:/Users/zk/Desktop/mmclassification-0.18.0/Dataset_ColorLine400/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')


# # optimizer1: imagenet_bs256
# optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', gamma=0.98, step=1)
# runner = dict(type='EpochBasedRunner', max_epochs=500)

# # optimizer2: imagenet_bs256_epochstep
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', step=[30, 60, 90])
# runner = dict(type='EpochBasedRunner', max_epochs=500)

# optimizer3: imagenet_bs256_200e_coslr_warmup
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=25025,
    warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=500)


# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# resume_from = None
resume_from = r'C:/Users/zk/Desktop/mmclassification-0.18.0/tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_300.pth'
workflow = [('train', 1), ('val', 1)]
