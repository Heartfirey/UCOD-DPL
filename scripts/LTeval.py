import os

from scripts.args import parse_train_args
import torch.multiprocessing as mp
from engine.runner.runner import Runner_local_refine
from engine.config.config import CfgNode

DATASET = ['CHAMELEON', 'TE-CAMO', 'TE-COD10K', 'NC4K']
def init_cfg(args) -> CfgNode:
    cfg = CfgNode.load_with_base(args.config)
    cfg = CfgNode(cfg)
    cfg.dataset_cfg.valset_cfg.keep_size=True
    cfg.train_cfg.checkpoint = args.load_from
    cfg.train_cfg.refiner_path = args.refiner_path
    cfg.mode = 'eval'
    cfg.work_dir = os.path.join(
        args.work_dir, 
        os.path.relpath(os.path.dirname(args.config),'./configs'),
        os.path.splitext(os.path.basename(args.config))[0]
    )
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.launcher = args.launcher
    return cfg
def main():
    args = parse_train_args()
    cfg = init_cfg(args)
    for dataset in DATASET:
        cfg.dataset_cfg.valset_cfg.DATASET = dataset
        print("running {}".format(dataset))
        runner = Runner_local_refine(cfg)
        runner.launch_val()

if __name__ == "__main__":
    
    main()
