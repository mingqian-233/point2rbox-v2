_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='Point2RBoxV2',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='Point2RBoxV2ResNet',
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
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=6,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        strides=[8],
        edge_loss_start_epoch=6,
        voronoi_type='standard',
        square_cls=[1, 9, 11],
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.001),
        loss_overlap=dict(
            type='GaussianOverlapLoss', loss_weight=10.0),
        loss_voronoi=dict(
            type='GaussianVoronoiLoss', loss_weight=5.0),
        loss_bbox_edg=dict(
            type='EdgeLoss', loss_weight=0.1),
        loss_bbox_syn=dict(
            type='RotatedIoULoss', loss_weight=1.0),
        loss_ss=dict(
            type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load point annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # Weakly supervised GTBox, (x,y,w,h,theta)
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 
        'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'ws_types'))
]

train_dataloader = dict(batch_size=2,
                        dataset=dict(pipeline=train_pipeline))

# e2e mode or pseudo generation mode. 
e2e_test_mode = True
if not e2e_test_mode:
    test_dataloader = _base_.val_dataloader
    test_evaluator = dict(_delete_=True,
                        type='DOTAMetric',
                        metric='mAP',
                        format_only=True,
                        outfile_prefix='data/split_ss_dota/point2rbox_v2_pseudo_labels')

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]
