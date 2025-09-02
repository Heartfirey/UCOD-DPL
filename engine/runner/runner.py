"""
USCOD Training Runner Module

This module provides a comprehensive training runner for USCOD (Unsupervised Camouflaged Object Detection)
with support for standard training and local refinement workflows.

Key Features:
- Standard USCOD training with baseline model and discriminator
- Local refinement training with sparse refiner
- Distributed training support via Accelerate
- Comprehensive checkpoint management
- LoRA adapter support
- Flexible configuration system
"""

import os
import warnings
import torch
from typing import Optional, Dict, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

# Core dependencies
from torch import nn
from accelerate import Accelerator
from safetensors.torch import load_file

# Model imports
from models.uscod import baseline
from models.discriminator import Discriminator
from models.UDLR import SparseRefiner

# Data imports - using new modular dataset structure
from data.datasets import DataLoaderFactory

# Engine imports
from engine.utils.logger import Logger
from engine.config import CfgNode

# Training loops
from .loop_UCOD_DPL import TrainLoop, ValLoop_Look_Twice
from .loop_CORAL import LocalRefineTrainLoop, LocalRefineValidationLoop
from .utils import extract_func_args

class BaseRunner(ABC):
    """
    Abstract base class for USCOD training runners.
    
    This class provides common functionality for all runner types including:
    - Accelerator setup and management
    - Logger initialization
    - Configuration management
    - Checkpoint saving/loading infrastructure
    """
    
    def __init__(self, config: CfgNode):
        """
        Initialize the base runner.
        
        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.logger: Optional[Logger] = None
        self.device = None
        self.start_epoch = 0
        
        # Initialize components in order
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all runner components in the correct order."""
        self.accelerator = self._build_accelerator()
        self.logger = self._build_logger()
        self.device = self.accelerator.device
        
        # Build model, optimizer, and dataloader (implemented by subclasses)
        self._build_model()
        self._build_optimizer()
        self._build_dataloader()
        
        # Prepare for distributed training
        self._prepare_accelerator()
        
        # Save configuration
        self._save_config_to_file()
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build the model(s). Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _build_optimizer(self) -> None:
        """Build optimizer(s) and scheduler(s). Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _build_dataloader(self) -> None:
        """Build data loaders. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _prepare_accelerator(self) -> None:
        """Prepare components for distributed training. Must be implemented by subclasses."""
        pass
    
    def _build_accelerator(self) -> Accelerator:
        """
        Build and configure the Accelerator for distributed training.
        
        Returns:
            Configured Accelerator instance
        """
        try:
            # Extract valid Accelerator arguments from config
            arg_list = extract_func_args(Accelerator)
            kwargs = {key: self.config[key] for key in arg_list if key in self.config.keys()}
            accelerator = Accelerator(**kwargs)
            return accelerator
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Accelerator: {e}")
    
    def _build_logger(self) -> Logger:
        """
        Build and configure the logger.
        
        Returns:
            Configured Logger instance
        """
        try:
            # Setup log path
            if self.config.exp_name is not None:
                log_path = os.path.join(self.config.work_dir, str(self.config.exp_name))
            else:
                timestamp = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
                log_path = os.path.join(self.config.work_dir, timestamp)
            
            os.makedirs(log_path, exist_ok=True)
            self.config.log_cfg.log_path = log_path
            return Logger(
                name=self.config.log_cfg.name,
                log_path=os.path.join(
                    log_path, 
                    f'{self.config.mode}{self.accelerator.process_index}.log'
                ),
                multi_rank=self.config.log_cfg.multi_rank
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Logger: {e}")
    
    def _save_config_to_file(self) -> None:
        """
        Save the configuration to a file in the log directory.
        """
        try:
            config_path = os.path.join(self.config.log_cfg.log_path, 'config.yaml')
            with open(config_path, 'w') as f:
                f.write(self.config.dump())
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def save_checkpoint(self, epoch: int, save_mode: str = 'model') -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            save_mode: Either 'model' for model only or 'all' for full state
        """
        save_path = self.config.log_cfg.log_path
        
        if save_mode == 'model':
            save_name = f'epoch{epoch}.pth'
            checkpoint_path = os.path.join(save_path, 'ckp', save_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_model(self.model, checkpoint_path)
                
        elif save_mode == 'all':
            save_name = f'epoch{epoch}'
            checkpoint_path = os.path.join(save_path, 'ckp', save_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load model checkpoint.
        
        First tries to load the latest checkpoint from log_path/ckp directory,
        then falls back to the provided checkpoint_path or config default.
        
        Args:
            checkpoint_path: Path to checkpoint, uses config default if None
        """
        self.start_epoch = 0
        checkpoint_path = self.config.train_cfg.get('checkpoint', None)
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint('ckp')
        try:
            checkpoint_path = os.path.join(checkpoint_path)
            state_dict = load_file(checkpoint_path, device="cuda")
            self.model.load_state_dict(state_dict)
            self.logger.info("Successfully loaded checkpoint weights from {}".format(checkpoint_path))
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
    
    def _find_latest_checkpoint(self, ckp_type) -> Optional[str]:
        """
        Find the latest checkpoint file in log_path/ckp directory.
        
        Returns:
            Path to the latest checkpoint file, or None if not found
        """
        try:
            if not hasattr(self.config, 'log_cfg') or not hasattr(self.config.log_cfg, 'log_path'):
                return None
                
            ckp_dir = os.path.join(os.path.dirname(self.config.log_cfg.log_path), ckp_type)
            if not os.path.exists(ckp_dir):
                return None
            
            # Find all checkpoint files
            checkpoint_files = []
            for file in os.listdir(ckp_dir):
                if file.endswith('.pth') or file.endswith('.pt'):
                    file_path = os.path.join(ckp_dir, file)
                    checkpoint_files.append(file_path)
            
            if not checkpoint_files:
                return None
            
            # Sort by modification time and return the latest
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            return latest_checkpoint
            
        except Exception as e:
            self.logger.warning(f"Error finding latest checkpoint: {e}")
            return None

class StandardRunner(BaseRunner):
    """
    Standard USCOD training runner with baseline model and discriminator.
    
    This runner implements the standard USCOD training workflow including:
    - Baseline model training
    - Discriminator training
    - Standard dataset loading
    - Checkpoint management
    """
    
    def __init__(self, config: CfgNode):
        """Initialize StandardRunner."""
        self.model = None
        self.discriminator = None
        self.optimizer = None
        self.dis_optimizer = None
        self.lr_scheduler = None
        self.dis_lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        super().__init__(config)
    
    def _build_model(self) -> None:
        """Build baseline model and discriminator."""
        try:
            self.model = baseline(self.config.model_cfg)
            self.discriminator = Discriminator(self.config.model_cfg)
            self.load_checkpoint()  # Load checkpoint after model creation
            self.logger.info("Successfully built baseline model and discriminator")
        except Exception as e:
            raise RuntimeError(f"Failed to build models: {e}")
    
    def _build_optimizer(self) -> None:
        """Build optimizers and schedulers for model and discriminator."""
        try:
            config = self.config.train_cfg
            
            # Model optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr0
            )
            
            # Discriminator optimizer
            self.dis_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=config.dis_lr0
            )
            
            # Learning rate schedulers
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.step_lr_size,
                gamma=config.step_lr_gamma
            )
            
            self.dis_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.dis_optimizer,
                step_size=config.dis_step_lr_size,
                gamma=config.dis_step_lr_gamma
            )
            
            self.logger.info("Successfully built optimizers and schedulers")
        except Exception as e:
            raise RuntimeError(f"Failed to build optimizers: {e}")
    
    def _build_dataloader(self) -> None:
        """Build standard data loaders using new modular dataset structure."""
        try:
            factory = DataLoaderFactory()
            
            # Build training dataloader
            self.train_dataloader = factory.create_train_loader(
                self.config.dataset_cfg,
            )
            
            # Build validation dataloader
            self.val_dataloader = factory.create_test_loader(
                self.config.dataset_cfg,
            )
            
            self.logger.info("Successfully built data loaders")
        except Exception as e:
            raise RuntimeError(f"Failed to build data loaders: {e}")
    
    def _prepare_accelerator(self) -> None:
        """Prepare all components for distributed training."""
        # Validate all components are initialized
        required_components = {
            'accelerator': self.accelerator,
            'model': self.model,
            'discriminator': self.discriminator,
            'optimizer': self.optimizer,
            'dis_optimizer': self.dis_optimizer,
            'train_dataloader': self.train_dataloader,
            'lr_scheduler': self.lr_scheduler,
            'dis_lr_scheduler': self.dis_lr_scheduler
        }
        
        for name, component in required_components.items():
            if component is None:
                raise RuntimeError(f"{name} has not been initialized")
        
        try:
            # Prepare components for distributed training
            (
                self.model,
                self.discriminator,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
                self.dis_optimizer,
                self.dis_lr_scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.discriminator,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
                self.dis_optimizer,
                self.dis_lr_scheduler
            )
            
            # Unwrap models if needed
            self.model = self.model.module if hasattr(self.model, "module") else self.model
            self.discriminator = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
            
            # Prepare validation dataloader
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
            
            self.logger.info("Successfully prepared components for distributed training")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare accelerator: {e}")
    
    def start_finetune(self):
        self._build_optimizer()

    def launch_train(self) -> None:
        """Launch the training loop."""
        try:
            self.trainloop = TrainLoop(self.config, self)
            self.trainloop.run()
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def launch_val_look_twice(self) -> Any:
        """Launch the look-twice validation loop."""
        try:
            loop = ValLoop_Look_Twice(self.config, self)
            return loop.run()
        except Exception as e:
            self.logger.error(f"Look-twice validation failed: {e}")
            raise


class LocalRefineRunner(BaseRunner):
    """
    Local refinement training runner with sparse refiner.
    
    This runner implements the local refinement training workflow including:
    - Baseline model (frozen)
    - Sparse refiner training
    - LR (Low-Resolution) dataset loading
    - Specialized checkpoint management for refiner
    """
    
    def __init__(self, config: CfgNode):
        """Initialize LocalRefineRunner."""
        self.model = None
        self.refiner = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        super().__init__(config)
    
    def _build_model(self) -> None:
        """Build baseline model and sparse refiner."""
        # try:
        self.model = baseline(self.config.model_cfg)
        self.refiner = SparseRefiner.from_config(self.config.model_cfg)
        
        # Freeze the baseline model
        self._freeze_model(self.model)
        
        self.load_checkpoint()  # Load checkpoint after model creation
        self.load_refiner_checkpoint()
        self.logger.info("Successfully built baseline model and sparse refiner")
        # except Exception as e:
        #     raise RuntimeError(f"Failed to build models: {e}")
    
    def _freeze_model(self, model: nn.Module) -> None:
        """Freeze a model by setting it to eval mode and disabling gradients."""
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.logger.info("Model frozen for local refinement training")
    
    def _build_optimizer(self) -> None:
        """Build optimizer and scheduler for refiner only."""
        try:
            config = self.config.train_cfg
            
            # Only optimize refiner parameters
            self.optimizer = torch.optim.AdamW(
                self.refiner.parameters(),
                lr=config.lr0
            )
            
            # Learning rate scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.step_lr_size,
                gamma=config.step_lr_gamma
            )
            
            self.logger.info("Successfully built refiner optimizer and scheduler")
        except Exception as e:
            raise RuntimeError(f"Failed to build optimizer: {e}")
    
    def _build_dataloader(self) -> None:
        """Build LR data loaders using new modular dataset structure."""
        # try:
        factory = DataLoaderFactory()
        window_size = getattr(self.config.model_cfg, 'window_size', None)
        
        # Build LR training dataloader
        self.train_dataloader = factory.create_lr_train_loader(
            self.config.dataset_cfg,
            window_size=window_size
        )
        
        # Build LR validation dataloader
        self.val_dataloader = factory.create_lr_test_loader(
            self.config.dataset_cfg,
            window_size=window_size
        )
        
        self.logger.info("Successfully built LR data loaders")
        # except Exception as e:
        #     raise RuntimeError(f"Failed to build LR data loaders: {e}")
    
    def _prepare_accelerator(self) -> None:
        """Prepare components for distributed training."""
        # Validate required components
        required_components = {
            'accelerator': self.accelerator,
            'model': self.model,
            'refiner': self.refiner,
            'optimizer': self.optimizer,
            'train_dataloader': self.train_dataloader,
            'lr_scheduler': self.lr_scheduler
        }
        
        for name, component in required_components.items():
            if component is None:
                raise RuntimeError(f"{name} has not been initialized")
        
        try:
            # Prepare components for distributed training
            (
                self.model,
                self.refiner,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.refiner,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler
            )
            
            # Unwrap models if needed
            self.model = self.model.module if hasattr(self.model, "module") else self.model
            self.refiner = self.refiner.module if hasattr(self.refiner, "module") else self.refiner
            
            # Prepare validation dataloader
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
            
            self.logger.info("Successfully prepared components for distributed training")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare accelerator: {e}")
    
    def save_checkpoint(self, epoch: int, save_mode: str = 'model') -> None:
        """
        Save refiner checkpoint.
        
        Args:
            epoch: Current epoch number
            save_mode: Either 'model' for refiner only or 'all' for full state
        """
        save_path = self.config.log_cfg.log_path
        
        if save_mode == 'model':
            save_name = f'epoch{epoch}.pth'
            checkpoint_path = os.path.join(save_path, 'refiner_ckp', save_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_model(self.refiner, checkpoint_path)
            
        elif save_mode == 'all':
            save_name = f'epoch{epoch}'
            checkpoint_path = os.path.join(save_path, 'refiner_ckp', save_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(checkpoint_path)
    
    def load_refiner_checkpoint(self, refiner_path: Optional[str] = None) -> None:
        """
        Load checkpoint with special handling for refiner.
        
        Args:
            checkpoint_path: Path to checkpoint, uses config default if None
        """
        if refiner_path == None:
            refiner_path = self.config.train_cfg.get('refiner_path', None)
        if refiner_path == None:
            refiner_path = self._find_latest_checkpoint('refiner_ckp')
        # Load refiner checkpoint if specified
        try:
            self.logger.info(f"Loading refiner from: {refiner_path}")
            refiner_model_path = os.path.join(refiner_path)
            state_dict = load_file(refiner_model_path, device="cuda")
            self.refiner.load_state_dict(state_dict, strict=True)
            self.logger.info("Successfully loaded refiner weights")
        except Exception as e:
            self.logger.error(f"Failed to load refiner: {e}")
    
    def launch_train(self) -> None:
        """Launch the local refinement training loop."""
        try:
            self.trainloop = LocalRefineTrainLoop(self.config, self)
            self.trainloop.run()
        except Exception as e:
            self.logger.error(f"Local refinement training failed: {e}")
            raise
    
    def launch_val(self) -> Any:
        """Launch the local refinement validation loop."""
        try:
            loop = LocalRefineValidationLoop(self.config, self)
            return loop.run()
        except Exception as e:
            self.logger.error(f"Local refinement validation failed: {e}")
            raise
    

# ============================================================================
# Backward Compatibility and Factory Functions
# ============================================================================

class RunnerFactory:
    """
    Factory class for creating runner instances.
    
    Provides automatic runner type detection and creation based on configuration.
    """
    
    _RUNNER_TYPES = {
        'standard': StandardRunner,
        'local_refine': LocalRefineRunner,
        'lr': LocalRefineRunner,  # Alias
    }
    
    @classmethod
    def create_runner(cls, config: CfgNode, runner_type: Optional[str] = None) -> BaseRunner:
        """
        Create a runner instance.
        
        Args:
            config: Configuration object
            runner_type: Explicit runner type, auto-detected if None
            
        Returns:
            Configured runner instance
        """
        if runner_type is None:
            runner_type = cls._detect_runner_type(config)
        
        if runner_type not in cls._RUNNER_TYPES:
            available = list(cls._RUNNER_TYPES.keys())
            raise ValueError(f"Unknown runner type '{runner_type}'. Available: {available}")
        
        runner_class = cls._RUNNER_TYPES[runner_type]
        return runner_class(config)
    
    @classmethod
    def _detect_runner_type(cls, config: CfgNode) -> str:
        """
        Automatically detect runner type from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Detected runner type
        """
        # Check for local refinement indicators
        if hasattr(config, 'model_cfg') and hasattr(config.model_cfg, 'window_size'):
            return 'local_refine'
        
        if hasattr(config, 'train_cfg') and config.train_cfg.get('refiner_path'):
            return 'local_refine'
        
        # Default to standard runner
        return 'standard'
    
    @classmethod
    def get_available_runners(cls) -> list:
        """Get list of available runner types."""
        return list(cls._RUNNER_TYPES.keys())


class Runner(StandardRunner):
    """
    Main Runner class for backward compatibility.
    
    This class extends StandardRunner to maintain compatibility with existing code
    while providing access to the new architecture.
    """
    
    def __init__(self, config: CfgNode):
        """
        Initialize Runner with backward compatibility warning.
        
        Args:
            config: Configuration object
        """
        warnings.warn(
            "Direct use of Runner class is deprecated. "
            "Use RunnerFactory.create_runner() or specific runner classes instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(config)


# Legacy alias for backward compatibility
Runner_local_refine = LocalRefineRunner


# Convenience functions
def create_runner(config: CfgNode, runner_type: Optional[str] = None) -> BaseRunner:
    """
    Create a runner instance using the factory pattern.
    
    Args:
        config: Configuration object
        runner_type: Optional explicit runner type
        
    Returns:
        Configured runner instance
    """
    return RunnerFactory.create_runner(config, runner_type)


def get_available_runner_types() -> list:
    """
    Get list of available runner types.
    
    Returns:
        List of available runner type names
    """
    return RunnerFactory.get_available_runners()


# Export main classes and functions
__all__ = [
    'BaseRunner',
    'StandardRunner',
    'LocalRefineRunner', 
    'Runner',  # Deprecated, but kept for compatibility
    'Runner_local_refine',  # Deprecated alias
    'RunnerFactory',
    'create_runner',
    'get_available_runner_types'
]


