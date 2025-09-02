import os

from scripts.args import parse_train_args
import torch.multiprocessing as mp
from engine.runner.runner import StandardRunner
from engine.config.config import CfgNode
from engine.utils.seed import set_random_seed

def init_cfg(args) -> CfgNode:
    cfg = CfgNode.load_with_base(args.config)
    cfg = CfgNode(cfg)
    cfg.dataset_cfg.valset_cfg.keep_size = False
    cfg.mode = 'train'
    cfg.work_dir = os.path.join(
        args.work_dir, 
        os.path.relpath(os.path.dirname(args.config),'./configs'),
        os.path.splitext(os.path.basename(args.config))[0]
    )
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.launcher = args.launcher
    return cfg

def main():
    set_random_seed(42)
    args = parse_train_args()
    
    cfg = init_cfg(args)
    runner = StandardRunner(cfg)
    runner.launch_train()

if __name__ == "__main__":
    main()
