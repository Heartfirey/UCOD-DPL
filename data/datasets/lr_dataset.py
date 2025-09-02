"""
LR Dataset implementation with patch-based processing.
"""
import torch
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from torchvision import transforms
from rich.progress import track

from engine.config import CfgNode
from .uscod_dataset import USCODDataset
from .transforms import ImageTransforms


class LRDataset(USCODDataset):
    """
    Low-Resolution dataset with patch-based feature extraction.
    Extends USCODDataset with windowed patch processing capabilities.
    """
    
    def __init__(
        self, 
        config: CfgNode, 
        feature_extractor_cfg: CfgNode, 
        mode: str, 
        dataset_dir: str, 
        cache_dir: str, 
        logger, 
        window_size: int = 3
    ):
        super().__init__(
            config=config,
            feature_extractor_cfg=feature_extractor_cfg, 
            dataset_dir=dataset_dir,
            cache_dir=cache_dir, 
            mode=mode, 
            logger=logger,
        )
        
        self.window_size = window_size
        self.require_m_patches = (mode == 'train' or config.require_m_patches)
        self.use_cache = config.use_cache
        
        # Setup grid and resize parameters
        self._setup_patch_parameters()
        
        # Setup patch transforms
        self._setup_patch_transforms()
        
        # Setup patch cache managers
        self._setup_patch_caches()
        
        # Prepare patch cache if needed
        if self._should_prepare_cache():
            self._prepare_patch_cache()
    
    def _setup_patch_parameters(self) -> None:
        """Setup patch-related parameters."""
        self.grid_h, self.grid_w = self.config.image_size
        self.resize_w = self.window_size * self.grid_h
        self.resize_h = self.window_size * self.grid_w
        image_size = (self.resize_w, self.resize_h)
        self.resize_img = transforms.Resize(image_size)
    
    def _setup_patch_transforms(self) -> None:
        """Setup patch-specific transforms."""
        self.patch_transform = ImageTransforms.get_patch_transform()
    
    def _setup_patch_caches(self) -> None:
        """Setup patch cache managers."""
        self.patch_cache = self.cache_manager.get_patch_cache()
        
        if self.require_m_patches:
            self.m_patch_cache = self.cache_manager.get_m_patch_cache()
        else:
            self.m_patch_cache = None
    
    def _should_prepare_cache(self) -> bool:
        """Check if cache preparation is needed."""
        return self.patch_cache.mode == 'w' or not self.use_cache
    
    def get_features(
        self, 
        img_path: str, 
        crop_center: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features from image patches.
        
        Args:
            img_path: Path to the image
            crop_center: Whether to crop center of image
            
        Returns:
            Tuple of (patches, m_patches)
        """
        self.prepare_feature_extractor()
        patches = []
        
        # Load and preprocess image
        image_ori = self.img_io.read_image(img_path, mode='RGB')
        if crop_center:
            image_ori = self._crop_center(image_ori)
        
        image = self.resize_img(image_ori)
        
        # Extract patch features
        patches = self._extract_patch_features(image)
        
        # Extract m_patches if required
        m_patches = None
        if self.require_m_patches:
            m_patches = self._extract_m_patch_features(image_ori)
        
        # Handle crop_center case
        if crop_center:
            img_tensor = self.transform_image(image_ori).cuda().unsqueeze(0)
            _, key = self.feature_extractor(img_tensor)
            return key, torch.stack(patches, dim=0).cuda().unsqueeze(0), m_patches

        return patches, m_patches
    
    def _crop_center(self, image: Image.Image) -> Image.Image:
        """Crop center portion of image."""
        original_width, original_height = image.size
        new_width = original_width // 2
        new_height = original_height // 2
        
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return image.crop((left, top, right, bottom))
    
    def _extract_patch_features(self, image: Image.Image) -> List[torch.Tensor]:
        """Extract features from image patches."""
        patches = []
        
        for i in range(self.window_size):
            for j in range(self.window_size):
                left = j * self.grid_w
                upper = i * self.grid_h
                right = left + self.grid_w
                lower = upper + self.grid_h
                
                patch = image.crop((left, upper, right, lower))
                patch_tensor = self.patch_transform(patch).unsqueeze(0).cuda()
                _, key = self.feature_extractor(patch_tensor)
                patches.append(key.squeeze(0).cpu())
        
        return patches
    
    def _extract_m_patch_features(self, image_ori: Image.Image) -> torch.Tensor:
        """Extract m-patch features."""
        image = self.feature_extractor_transform(image_ori).unsqueeze(0).cuda()
        _, key = self.feature_extractor(image)
        
        patch_tensor = []
        for i in range(2):
            for j in range(2):
                left = j * 18
                upper = i * 18
                right = left + 36
                lower = upper + 36
                patch_tensor.append(key[:, :, upper:lower, left:right])
        
        return torch.stack(patch_tensor, dim=1)
    
    def _prepare_patch_cache(self) -> None:
        """Prepare patch cache."""
        self.prepare_feature_extractor()
        self.patches = []
        self.m_patches = []

        for img_path in track(
            self.image_paths, 
            description=f'Divide the image into windows by {self.feature_extractor_cfg.backbone}'
        ):
            patches, m_patches = self.get_features(img_path)
            self.patches.append(torch.stack(patches, dim=0).cpu())
            
            if self.require_m_patches and m_patches is not None:
                self.m_patches.extend([
                    each.clone().detach().cpu() for each in m_patches
                ])

        # Cache patches
        if self.use_cache:
            self.patch_cache.dump_list(self.patches)
            
            if self.require_m_patches and self.m_patch_cache:
                self.m_patch_cache.dump_list(self.m_patches)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get dataset item with patch features."""
        items = super().__getitem__(index)
        
        # Load patch features
        if self.use_cache:
            h_inputs = self.patch_cache.read_file(index)
            m_inputs = None
            if self.require_m_patches and self.m_patch_cache:
                m_inputs = self.m_patch_cache.read_file(index)
        else:
            h_inputs = self.patches[index]
            m_inputs = None
            if self.require_m_patches:
                m_inputs = self.m_patches[index]
        
        items.update({
            'm_inputs': m_inputs,
            'h_inputs': h_inputs,
            'index': [index]
        })
        
        return items
