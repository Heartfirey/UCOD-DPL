"""
DataLoader utilities and collate functions.
"""
import torch
from typing import Any, Dict, List
from torch.utils.data import DataLoader

from engine.config import CfgNode
from .uscod_dataset import USCODDataset
from .lr_dataset import LRDataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching dataset items.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched dictionary
    """
    batch_dict = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if any(v is None for v in values):
            batch_dict[key] = values
        else:
            try:
                if isinstance(values[0], torch.Tensor):
                    batch_dict[key] = torch.stack(values)
                else:
                    batch_dict[key] = torch.tensor(values)
            except Exception:
                batch_dict[key] = values

    return batch_dict


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def create_train_loader(
        config: CfgNode, 
        logger=None, 
    ) -> DataLoader:
        """Create training data loader for USCOD dataset."""
        dataset = USCODDataset(
            config=config.trainset_cfg,
            feature_extractor_cfg=config.feature_extractor_cfg,
            mode='train',
            dataset_dir=config.dataset_dir,
            cache_dir=config.cache_dir,
            logger=logger,
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.trainloader_cfg.batch_size,
            num_workers=config.trainloader_cfg.num_workers,
            shuffle=config.trainloader_cfg.shuffle,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        if logger:
            logger.log(
                f"{len(train_loader)} batches of train dataloader "
                f"{config.trainset_cfg.DATASET} has been created."
            )
        
        return train_loader
    
    @staticmethod
    def create_test_loader(
        config: CfgNode, 
        logger=None, 
    ) -> DataLoader:
        """Create test data loader for USCOD dataset."""
        dataset = USCODDataset(
            config=config.valset_cfg,
            feature_extractor_cfg=config.feature_extractor_cfg,
            mode='test',
            dataset_dir=config.dataset_dir,
            cache_dir=config.cache_dir,
            logger=logger,
        )
        
        test_loader = DataLoader(
            dataset,
            num_workers=config.val_loader_cfg.num_workers,
            batch_size=config.val_loader_cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        if logger:
            logger.log(
                f"{len(test_loader)} batches of test dataloader "
                f"{config.valset_cfg.DATASET} has been created."
            )
        
        return test_loader
    
    @staticmethod
    def create_lr_train_loader(
        config: CfgNode, 
        logger=None, 
        window_size: int = 3
    ) -> DataLoader:
        """Create training data loader for LR dataset."""
        dataset = LRDataset(
            config=config.trainset_cfg,
            feature_extractor_cfg=config.feature_extractor_cfg,
            mode='train',
            dataset_dir=config.dataset_dir,
            cache_dir=config.cache_dir,
            logger=logger,
            window_size=window_size,
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.trainloader_cfg.batch_size,
            num_workers=config.trainloader_cfg.num_workers,
            shuffle=config.trainloader_cfg.shuffle,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        if logger:
            logger.log(
                f"{len(train_loader)} batches of train dataloader "
                f"{config.trainset_cfg.DATASET} has been created."
            )
        
        return train_loader
    
    @staticmethod
    def create_lr_test_loader(
        config: CfgNode, 
        logger=None, 
        window_size: int = 3
    ) -> DataLoader:
        """Create test data loader for LR dataset."""
        dataset = LRDataset(
            config=config.valset_cfg,
            feature_extractor_cfg=config.feature_extractor_cfg,
            mode='test',
            dataset_dir=config.dataset_dir,
            cache_dir=config.cache_dir,
            logger=logger,
            window_size=window_size,
        )
        
        test_loader = DataLoader(
            dataset,
            num_workers=config.val_loader_cfg.num_workers,
            batch_size=config.val_loader_cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        if logger:
            logger.log(
                f"{len(test_loader)} batches of test dataloader "
                f"{config.valset_cfg.DATASET} has been created."
            )
        
        return test_loader


# Backward compatibility functions
def init_trainloader(config: CfgNode, logger=None) -> DataLoader:
    """Initialize training data loader (backward compatibility)."""
    return DataLoaderFactory.create_train_loader(config, logger)


def init_testloaders(config: CfgNode, logger=None) -> DataLoader:
    """Initialize test data loader (backward compatibility)."""
    return DataLoaderFactory.create_test_loader(config, logger)


def init_trainloader_LR(
    config: CfgNode, 
    logger=None, 
    window_size: int = 3
) -> DataLoader:
    """Initialize LR training data loader (backward compatibility)."""
    return DataLoaderFactory.create_lr_train_loader(config, logger, window_size)


def init_testloaders_LR(
    config: CfgNode, 
    logger=None, 
    window_size: int = 3
) -> DataLoader:
    """Initialize LR test data loader (backward compatibility)."""
    return DataLoaderFactory.create_lr_test_loader(config, logger, window_size)
