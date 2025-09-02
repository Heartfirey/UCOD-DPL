"""
Image transformation utilities for COD datasets.
"""
from typing import Tuple
from torchvision import transforms


class ImageTransforms:
    """Centralized image transformation configurations."""
    
    @staticmethod
    def get_image_transform(image_size: Tuple[int, int] = (518, 518)) -> transforms.Compose:
        """Get standard image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def get_label_transform(image_size: Tuple[int, int] = (518, 518), keep_size=False) -> transforms.Compose:
        """Get label transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ][keep_size:])
    
    @staticmethod
    def get_feature_extractor_transform(img_size) -> transforms.Compose:
        """Get feature extractor transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(img_size),  # 2/3 dinov2
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def get_patch_transform() -> transforms.Compose:
        """Get patch transformation pipeline."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
