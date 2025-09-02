cfg = dict(
    dataset_cfg = dict(
        cache_dir='./datasets/cache/look_twice',
        dataset_dir='/data/ywq/dataset/RefCOD',
        trainset_cfg = dict(
            DATASET='TR-CAMO+TR-COD10K',
            require_label=False,
        ),
    valset_cfg = dict(
            DATASET='TE-COD10K',
            require_label=True,
        ),
    )     
)