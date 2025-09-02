import os
import sys
import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='config file path', required=True)
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work dir')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--load_from', type=str, default=None, help='load from checkpoint')
    parser.add_argument('--refiner_path', type=str, default=None, help='load refiner checkpoint')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args

def parse_test_args():
    pass
