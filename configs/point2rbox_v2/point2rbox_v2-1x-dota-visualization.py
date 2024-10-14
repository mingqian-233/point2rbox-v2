_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='Point2RBoxV2',
    debug=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=7,
        out_indices=(1, 2, 3, 4, 5, 6),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048, 2048, 2048, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=6,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=512,
        feat_channels=512,
        strides=[8],
        square_cls=[1, 9, 11],
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=False,
            num_step=3,
            thr_mod=0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # Weakly supervised GTBox, (x,y,w,h,theta)
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # dict(
    #     type='mmdet.RandomFlip',
    #     prob=0.75,
    #     direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 
        'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'ws_types'))
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='train-p2r/annfiles/',
        data_prefix=dict(img_path='train-p2r/images/'),
        pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=72,
        by_epoch=True,
        milestones=[48, 66],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=72, val_interval=72)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]
load_from = 'epoch_12.pth'
