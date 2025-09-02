cfg = dict(
    _BASE_ = [
        './UCOD-DPL_dinov1.py'
    ],
    start_ema = 1,
    enable_plabel_cache=True,
    train_cfg = dict(
        max_epoch=8,
        lr0=2e-4,
        step_lr_size=2,
        step_lr_gamma=0.95,
    ),
    val_cfg = dict(
        val_interval=4,
        val_start = 4
    ),
    model_cfg=dict(
        window_size = 3,
        window_length=56,
        threshold = 0.0015,
        ema_weight = 0.70,
    ),
    dataset_cfg=dict(
        # cache_dir='./datasets/cache/dinov1',
        trainloader_cfg = dict(
            batch_size=2,
            num_workers=0,
            shuffle=True
        ),
        valset_cfg = dict(
            use_cache=True,
            require_m_patches=True,
        ),
        trainset_cfg = dict(
            look_twice = False,
            image_size=(296,296),
            require_label=True,
            look_twice_th=0.15,
            bkg_th=0.6,
            use_cache=True,
            require_m_patches=True,
        ),
    )
)
