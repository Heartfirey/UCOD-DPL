from .config import CfgNode

#* Deep Leanring Basic Config
BaseCfg = dict(
    #* Model
    model = dict(),
    #* Dataset
    dataset = dict(),
    #* Optimizer
    optimizer = dict(),
    #* Scheduler
    scheduler = dict(),
    #* Trainer
    runner = dict(),
    #* Logger
    logger = dict(),
    #* Hook
    hook = dict(),
    #* Loop
    loop = dict()
)

LoopBaseCfg = dict(
    mode=['train', 'val', 'test'][0],
    max_epoch=0,
    start_epoch=0,
    
)


