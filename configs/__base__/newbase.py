
cfg = dict(
    work_dir='./work',
    train_cfg = dict(
        dist_train=True,
        max_epoch=25,
        start_finetune=-5,
        merge_alpha=0.5,
        start_epoch=0,
        merge_method="dis",
        add_noise=False,
        grad_norm=1.0,
        save_cfg = dict(
            save_mode = ['model', 'all'][0],
            save_interval = 5,
            start_save = -50,
        ),
    ),
    model_cfg=dict(
        decoder='BGDecoder',
        up_sample=False,
        dis_use_features=True,
        feature_size=16,
        ema_weight = 0.999,
        dim=768,
        use_attention=False,
        conv_num=1
    ),
    val_cfg = dict(
        enable_val = True,
        val_interval = 5,
        start_val = -50,
    ),
    log_cfg = dict(
        name = "Ablation 1",
        log_path = "/home/yanweiq/storage/trainlog.log",
        multi_rank=[0]
    ),
    dataset_cfg=dict(
        trainset_cfg = dict(
            type='USCODDataset',
            
        ),
        trainloader_cfg = dict(
            
        ),
        valset_cfg = dict(
            type='USCODDataset',
            
        ),
        val_loader_cfg = dict(
            
        ),
    ),
        
    feature_extractor_cfg=dict(
        
    )
)
