"""
USCOD Dataset implementation.
"""
from typing import Dict, Any
from engine.config import CfgNode
from .base_dataset import BaseCODDataset


class USCODDataset(BaseCODDataset):
    """
    Dataset for Unsupervised Camouflaged Object Detection (USCOD).
    Containing the image feature and the pseudo label building process.
    """
    
    def __init__(
        self, 
        config: CfgNode, 
        feature_extractor_cfg: CfgNode, 
        mode: str, 
        dataset_dir: str, 
        cache_dir: str, 
        logger, 
    ):
        super().__init__(
            config=config,
            feature_extractor_cfg=feature_extractor_cfg, 
            dataset_dir=dataset_dir,
            cache_dir=cache_dir, 
            mode=mode, 
            load_all=mode == 'test',
            image_size=config.image_size,
            require_label=config.require_label, 
            logger=logger,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item."""
        return super().__getitem__(index)
