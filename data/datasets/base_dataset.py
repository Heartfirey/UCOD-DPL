"""
Base dataset class for COD feature extraction.
"""
import os
import torch
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from engine.config import CfgNode
from engine.utils.fileio.backend import ImageIO
from data.utils.found_bkg_mask import compute_img_bkg_seg
from data.utils.feature_extractor import backbone
from rich.progress import track

from .transforms import ImageTransforms
from .cache_manager import MultiCacheManager


class BaseCODDataset(Dataset):
    """Base class for COD feature datasets."""
    
    def __init__(
        self,
        config: CfgNode,
        feature_extractor_cfg: CfgNode,
        dataset_dir: str,
        cache_dir: str = None,
        mode: str = 'train',
        load_all: bool = False,
        keep_size: bool = False,
        image_size: Tuple[int, int] = (518, 518),
        require_label: bool = False,
        logger=None,
    ):
        self.config = config
        self.feature_extractor_cfg = feature_extractor_cfg
        self.mode = mode
        self.cache_dir = cache_dir
        self.logger = logger
        self.load_all = load_all
        self.keep_size = keep_size
        self.image_size = image_size
        self.require_label = require_label
        
        # Initialize transforms
        self._setup_transforms()
        
        # Initialize file paths
        self._setup_file_paths(dataset_dir)
        
        # Initialize cache managers
        self._setup_cache_managers()
        
        # Prepare cache if needed
        if self.cache_manager.get_features_cache().mode == 'w':
            self._prepare_cache()
    
    def _setup_transforms(self) -> None:
        """Setup image transformations."""
        self.transform_image = ImageTransforms.get_image_transform(self.image_size) 
        self.transform_label = ImageTransforms.get_label_transform(self.image_size, self.load_all or self.keep_size)
    
    def _setup_file_paths(self, dataset_dir: str) -> None:
        """Setup image and label file paths."""
        self.img_io = ImageIO(backend='PIL')
        self.image_paths = []
        self.label_paths = []
        
        for dataset in self.config.DATASET.split('+'):
            image_dir = os.path.join(dataset_dir, dataset, 'im')
            label_dir = os.path.join(dataset_dir, dataset, 'gt')
            self.image_paths.extend(self.img_io.list_dir_image(image_dir))
            if self.require_label:
                self.label_paths.extend(self.img_io.list_dir_image(label_dir))
        
        self.image_paths = sorted(self.image_paths)
        if self.label_paths:
            self.label_paths = sorted(self.label_paths)
            
        if self.require_label:
            self._check_file_mapping(self.image_paths, self.label_paths)
    
    def _setup_cache_managers(self) -> None:
        """Setup cache managers."""
        self.cache_manager = MultiCacheManager(
            cache_dir=self.cache_dir,
            feature_extractor_type=self.feature_extractor_cfg.type,
            mode=self.mode,
            dataset_name=self.config.DATASET,
            logger=self.logger
        )
    
    def _check_file_mapping(self, list_a: List[Path], list_b: List[Path]) -> None:
        """Check file mapping between image and label lists."""
        assert len(list_a) == len(list_b), "Length of two lists should be the same"
        
        stem_a = [each.stem for each in list_a]
        stem_b = [each.stem for each in list_b]
        for each_a_stem in stem_a:
            assert each_a_stem in stem_b, f"File {each_a_stem} not found in list_b"
    
    def prepare_feature_extractor(self) -> None:
        """Prepare feature extractor if not already initialized."""
        if not hasattr(self, "feature_extractor_transform"):
            if self.feature_extractor_cfg.type == 'dinov1':
                img_size = (432, 432)
            elif self.feature_extractor_cfg.type == 'dinov2':
                img_size = (756, 756)
            self.feature_extractor_transform = ImageTransforms.get_feature_extractor_transform(img_size)
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = backbone(self.feature_extractor_cfg)
    
    def _get_features(
        self, 
        img_tensor: torch.Tensor, 
    ) -> torch.Tensor:
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        _, key = self.feature_extractor(img_tensor)
        return key
    
    def _prepare_cache(self) -> None:
        """Prepare feature cache."""
        self.prepare_feature_extractor()
        all_feature_list = []
        all_plabel_list = []
        
        for img_path in track(
            self.image_paths, 
            description=f'Extracting image features by {self.feature_extractor_cfg.backbone}'
        ):
            image = self.img_io.read_image(img_path, mode='RGB')
            img = self.transform_image(image).unsqueeze(0).cuda()
            
            features = self._get_features(img)
            all_feature_list.append(features.squeeze(0).to('cpu'))
        
        # Cache features
        features_cache = self.cache_manager.get_features_cache()
        features_cache.dump_list(all_feature_list)
        if self.mode == 'train':
            pseudo_label_cache = self.cache_manager.get_pseudo_label_cache()
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item."""
        img_path = self.image_paths[index]
        
        # Load label if required
        label_tensor = None
        if self.label_paths:
            label_tensor = self.img_io.read_image(self.label_paths[index], 'L')
            label_tensor = self.transform_label(label_tensor)
        
        # Load features from cache
        features = None
        features_cache = self.cache_manager.get_features_cache()
        if features_cache:
            features = features_cache.read_file(index)
        
        # Load pseudo label if available
        pseudo_label = None
        pseudo_label_cache = self.cache_manager.get_pseudo_label_cache()
        if pseudo_label_cache:
            pseudo_label = pseudo_label_cache.read_file(index)
        
        return {
            "pseudo_label": pseudo_label,
            "label_tensor": label_tensor,
            "features": features,
            "img_path": str(img_path)
        }
