import inspect
import torch
def extract_func_args(func: callable) -> list:
    sig = inspect.signature(func)
    return sig.parameters
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from accelerate import Accelerator

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

_EPS = np.spacing(1)

class ProgressManager:
    def __init__(self, accelerator: Accelerator):
        """
        进度条管理类，确保在 DDP 环境下仅在主进程（rank=0）运行 `rich`
        """
        self.accelerator = accelerator
        self.progress = None
        self.tasks = {}

    def setup_progress(self):
        """仅在主进程（rank=0）创建 `rich` 进度条"""
        if self.accelerator.is_main_process:
            self.progress = Progress(
                SpinnerColumn(), 
                TextColumn("[progress.description]{task.description}"), 
                BarColumn(), 
                TaskProgressColumn(), 
                TimeRemainingColumn(), 
                TextColumn("[bold orange]Time Elapsed:[/bold orange]"),
                TimeElapsedColumn(),
                expand=False
            )

    def add_task(self, name, total):
        """添加进度条任务"""
        if self.accelerator.is_main_process and self.progress:
            self.tasks[name] = self.progress.add_task(f"[bold]{name}[/bold]", total=total, start=False)

    def start_task(self, name):
        """启动指定进度任务"""
        if self.accelerator.is_main_process and self.progress and name in self.tasks:
            self.progress.start_task(self.tasks[name])

    def update_task(self, name, advance=1):
        """更新进度条"""
        if self.accelerator.is_main_process and self.progress and name in self.tasks:
            self.progress.update(self.tasks[name], advance=advance)

    def reset_task(self, name):
        """重置进度条"""
        if self.accelerator.is_main_process and self.progress and name in self.tasks:
            self.progress.reset(self.tasks[name], start=False)

    def __enter__(self):
        """进入 `with` 语句时，启动 `rich` 进度条（仅在主进程）"""
        if self.accelerator.is_main_process and self.progress:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出 `with` 语句时，结束 `rich` 进度条（仅在主进程）"""
        if self.accelerator.is_main_process and self.progress:
            self.progress.__exit__(exc_type, exc_value, traceback)