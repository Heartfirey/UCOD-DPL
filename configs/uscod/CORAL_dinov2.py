cfg = dict(
    _BASE_ = [
        './UCOD-DPL_dinov2.py'
    ],
    start_ema = 1,
    enable_plabel_cache=True,
    train_cfg = dict(
        max_epoch=8,
        lr0=1e-4,
        step_lr_size=2,
        step_lr_gamma=0.95,
    ),
    val_cfg = dict(
        val_interval=4,
        val_start=4,
    ),
    model_cfg=dict(
        window_size = 3,
        window_length=56,
        threshold = 0.0015,
        ema_weight = 0.70,
    ),
    dataset_cfg=dict(
        # cache_dir='./datasets/cache/dinov2',
        trainloader_cfg = dict(
            batch_size=2,
            num_workers=0,
            shuffle=True
        ),
        valset_cfg = dict(
            DATASET='TE-CAMO',
            use_cache=True,
            require_m_patches=False,
        ),
        trainset_cfg = dict(
            look_twice = False,
            image_size=(518,518),
            require_label=True,
            look_twice_th=0.15,
            bkg_th=0.6,
            use_cache=True,
            require_m_patches=True,
        ),
    )
)
