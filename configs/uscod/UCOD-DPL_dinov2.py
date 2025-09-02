cfg = dict(
    _BASE_ = [
        '../__base__/accelerate.py',
        '../__base__/newbase.py',
        '../dataset/cod4040.py'
    ],
    exp_name = 'UCOD-DPL_dinov2',
    train_cfg = dict(
        max_epoch=25,
        dis_epoch=1,
        start_epoch=0,
        lr0=2e-4,
        dis_lr0=1e-3,
        dis_intertrain=2,
        step_lr_size=25,
        dis_step_lr_size=25,
        step_lr_gamma=0.95,
        dis_step_lr_gamma=0.95,
    ),
    val_cfg = dict(
        look_twice=True,
        look_twice_th=0.15,
        expand_type='dynamic',
        val_interval = 5,
        val_start = 5,
    ),
    log_cfg = dict(
        log_interval=50,
    ),
    model_cfg=dict(
        ema_weight = 0.99,
        dim=768,
        dis_use_features=False,
        feature_size=68,
    ),
    dataset_cfg=dict(
        cache_dir='./datasets/cache',
        val_loader_cfg = dict(
            batch_size=1,
            num_workers=0,
            shuffle=False
        ),
        trainloader_cfg = dict(
            batch_size=16,
            num_workers=0,
            shuffle=True
        ),
        valset_cfg = dict(
            DATASET='TE-CAMO',
            require_label=True,
            image_size=(518,518),
        ),
        trainset_cfg = dict(
            DATASET='TR-CAMO+TR-COD10K',
            image_size=(518,518),
            require_label=False,
            bkg_th=0.6,
        ),
        feature_extractor_cfg=dict(
            type='dinov2',
            backbone_weight_base = '~/workspace/weights/huggingface',
            backbone = 'facebook/dinov2-base',
            backbone_weights = './weights',
            backbone_type = 'huggingface',
            backbone_feat_dim=[768],
        ),
    )
)
