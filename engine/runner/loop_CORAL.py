"""
Local Refine Loop Module for USCOD Training and Validation

This module implements the training and validation loops for local refinement
in the USCOD (Unsupervised Salient Object Detection) framework.
"""

# Standard library imports
import os
import math
import random
from typing import Any, Tuple, Optional, Dict, List

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from rich.progress import track
from cv2 import connectedComponents, boundingRect
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

# Local imports
from engine.config.config import CfgNode
from engine.utils.metrics.metric import calculate_cod_metrics, statistics
from engine.utils.fileio.backend import ImageIO, TorchPickleIO
from engine.runner.loop_UCOD_DPL import BaseLoop
from engine.utils.show_imgs import draw_bboxes_on_image_and_save
from data.utils.feature_extractor import backbone
from .utils import ProgressManager
from ..utils.save_image import  save_tensor_binary_mask_as_image

class LocalRefineTrainLoop(BaseLoop):
    pass

class LocalRefineValidationLoop(BaseLoop):
    
    def __init__(self, config: CfgNode, runner):
        """
        Initialize the validation loop.
        
        Args:
            config: Configuration node containing validation parameters
            runner: Runner instance containing model, dataloader, etc.
        """
        super().__init__(config, runner)
        self._mode = 'val'
        self._runner = runner
        self.window_length = self.cfg.model_cfg.window_length
        
        # Initialize progress tracking
        self.progress_manager = ProgressManager(self.runner.accelerator)
        self.progress_manager.setup_progress()
        self.progress_manager.add_task("Validation Iteration", total=len(self.runner.val_dataloader))

    #TODO:参数化    
    def concate_preds(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Concatenate patch predictions into a full mask.
        
        This method combines predictions from 4 overlapping patches (2x2 grid)
        into a single full-resolution mask by averaging overlapping regions.
        
        Args:
            preds: Patch predictions tensor of shape (batch, 4, channels, 68, 68)
            
        Returns:
            Full mask tensor of shape (batch, channels, 102, 102)
        """
        with torch.no_grad():
            b, n, c, h, w = preds.shape
            
            # Initialize full mask and overlap counter
            full_mask = torch.zeros((b, c, 102, 102), device=preds.device)
            counter = torch.zeros((b, c, 102, 102), device=preds.device)
            
            # Combine patches in 2x2 grid pattern
            for i in range(2):
                for j in range(2):
                    left = j * 34
                    upper = i * 34
                    right = left + 68
                    lower = upper + 68
                    
                    patch_idx = i * 2 + j
                    full_mask[:, :, upper:lower, left:right] += preds[:, patch_idx, :, :, :]
                    counter[:, :, upper:lower, left:right] += 1.0
            
            # Average overlapping regions
            full_mask = full_mask / (counter + 1e-6)
            return full_mask

    def run(self) -> Dict[str, float]:
        """
        Run validation loop and return metrics.
        
        Returns:
            Dictionary containing validation metrics
        """
        statistics_val = statistics()
        self.runner.refiner.eval()
        self.progress_manager.start_task("Validation Iteration")
        
        # Log validation start
        if hasattr(self.runner, "trainloop"):
            epoch_info = f'{self.runner.trainloop._cur_epoch}/{self.runner.trainloop._max_epoch}'
            self.runner.logger.log(f'[green]epoch:{epoch_info}:start validate[/green]')
        else:
            dataset_name = self.cfg.dataset_cfg.valset_cfg.DATASET
            self.runner.logger.log(f'[green]start validate on {dataset_name}[/green]')
        
        # Process all validation batches
        for batch in tqdm(self.runner.val_dataloader, desc="Validating"):
            self._process_validation_batch(batch, statistics_val)
            self.progress_manager.update_task("Validation Iteration")
        
        # Get and log results
        result = statistics_val.get_result()
        result_table = {key: [round(result[key], 4)] for key in result.keys()}
        self.runner.logger.log_table(result_table)
        self.progress_manager.reset_task("Validation Iteration")
        
        return result  
    
    def _process_validation_batch(self, batch: Dict[str, torch.Tensor], 
                                 statistics_val: statistics) -> None:
        """
        Process a single validation batch.
        
        Args:
            batch: Batch data dictionary
            statistics_val: Statistics accumulator
        """
        _, label_tensor, l_input_features, img_path, m_input_features, h_input_features, _ = batch.values()
        
        with torch.no_grad():
            # Prepare features and get predictions
            features_dict = self._prepare_validation_features(
                l_input_features, m_input_features, h_input_features
            )
            
            # Check if center cropping is needed
            crop_center = self._should_crop_center(features_dict['preds'])
            if crop_center:
                features_dict = self.crop_center(img_path[0])
            
            # Get refiner outputs
            outputs, _, opt = self.runner.refiner(
                features_dict['l_features'],
                features_dict['h_features'], 
                features_dict['preds']
            )
            
            # Gather predictions across devices
            all_preds, all_labels = self.runner.accelerator.gather_for_metrics((outputs, label_tensor))
            if crop_center:
                all_preds = self._center_pad(all_preds)
            # Process and save predictions
            preds_up = self.process_preds(all_preds, all_labels.shape[2:])
            self._save_prediction_image(preds_up, 'preds', img_path[0])
            statistics_val.step(all_labels, preds_up)
    
    def _center_pad(self, input_tensor: torch.Tensor, fill_value: float = -10.0) -> torch.Tensor:
        if input_tensor.dim() == 3:
            c, h, w = input_tensor.shape
            batch_dim = False
        elif input_tensor.dim() == 4:
            b, c, h, w = input_tensor.shape
            batch_dim = True
        else:
            raise ValueError("shape error:{}".format(input_tensor.shape))

        padded_h, padded_w = h * 2, w * 2

        if batch_dim:
            padded_output_tensor = torch.full(
                (b, c, padded_h, padded_w),
                fill_value,
                device=input_tensor.device,
                dtype=input_tensor.dtype
            )
        else:
            padded_output_tensor = torch.full(
                (c, padded_h, padded_w),
                fill_value,
                device=input_tensor.device,
                dtype=input_tensor.dtype
            )

        offset_h = h // 2
        offset_w = w // 2


        if batch_dim:
            padded_output_tensor[:, :, offset_h:offset_h + h, offset_w:offset_w + w] = input_tensor
        else:
            padded_output_tensor[:, offset_h:offset_h + h, offset_w:offset_w + w] = input_tensor

        return padded_output_tensor
    
    def _prepare_validation_features(self, l_input_features: torch.Tensor, 
                                   m_input_features: torch.Tensor,
                                   h_input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare features for validation.
        
        Args:
            l_input_features: Low-resolution features
            m_input_features: Medium-resolution features  
            h_input_features: High-resolution features
            
        Returns:
            Dictionary containing prepared features and predictions
        """
        h = w = self.window_length
        b, c, _, _ = l_input_features.shape
        
        # Interpolate features
        l_features = F.interpolate(l_input_features, size=(h, w), mode='bilinear')
        h_features = F.interpolate(
            h_input_features.flatten(0, 1), size=(h, w), mode='bilinear'
        ).reshape(b, -1, c, h, w)
        
        # Get predictions based on configuration
        if self.cfg.dataset_cfg.valset_cfg.require_m_patches:
            m_features = F.interpolate(m_input_features.flatten(0, 1), size=(68, 68), mode='bilinear')
            preds, _, _ = self.runner.model(m_features)
            preds = self.concate_preds(preds.reshape(b, -1, 1, 68, 68))
        else:
            preds, _, _ = self.runner.model(l_features)
                    # Apply center padding if cropping was used
        # if crop_center:
        #     all_preds = self.center_pad_with_value(all_preds)
        #     opt['h_preds'] = self.center_pad_with_value(opt['h_preds'])
        #     opt['preds'] = self.center_pad_with_value(opt['preds'])
        return {
            'l_features': l_features,
            'h_features': h_features,
            'preds': preds
        }
    
    def _should_crop_center(self, preds: torch.Tensor) -> bool:
        """
        Determine if center cropping should be applied based on prediction density.
        
        Args:
            preds: Prediction tensor
            
        Returns:
            True if center cropping should be applied
        """
        positive_ratio = (preds > 0).sum() / (preds.shape[2] * preds.shape[3])
        return positive_ratio < 0.001
    
    def _save_prediction_image(self, preds_up: torch.Tensor, img_type:str, img_path: str) -> None:
        """
        Save prediction as binary mask image.
        
        Args:
            preds_up: Upsampled predictions
            img_path: Original image path for naming
        """
        save_dir = os.path.join(
            self.cfg.log_cfg.log_path, 
            img_type, 
            self.cfg.dataset_cfg.valset_cfg.DATASET
        )
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        save_tensor_binary_mask_as_image(preds_up[0] > 0.5, save_path)
    
    def crop_center(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Crop center region and recompute features for low-confidence predictions.
        
        Args:
            path: Image path
            
        Returns:
            Dictionary containing cropped features and predictions
        """
        # Get cropped features from dataset
        l_input_features, h_input_features, m_input_features = \
            self.runner.val_dataloader.dataset.get_features(path, crop_center=True)
        
        h = w = self.window_length
        b, c, _, _ = l_input_features.shape
        
        # Interpolate features to target size
        l_features = F.interpolate(l_input_features, size=(h, w), mode='bilinear')
        h_features = F.interpolate(
            h_input_features.flatten(0, 1), size=(h, w), mode='bilinear'
        ).reshape(b, -1, c, h, w)
        
        # Get predictions based on configuration
        if self.cfg.dataset_cfg.valset_cfg.require_m_patches:
            m_features = F.interpolate(m_input_features.flatten(0, 1), size=(68, 68), mode='bilinear')
            preds, _, _ = self.runner.model(m_features)
            preds = self.concate_preds(preds.reshape(b, -1, 1, 68, 68))
        else:
            preds, _, _ = self.runner.model(l_features)
        
        return {
            'l_features': l_features,
            'h_features': h_features,
            'preds': preds
        }

    def process_preds(self, preds: torch.Tensor, 
                    size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process predictions by applying sigmoid, computing scores, and upsampling.
        
        Args:
            preds: Raw predictions tensor
            size: Target size (height, width) for upsampling
            binary: Whether to apply binary thresholding (unused but kept for compatibility)
            
        Returns:
            Tuple of (upsampled_predictions, confidence_scores)
        """
        h, w = size
        
        # Apply sigmoid if not already applied
        if torch.all((preds >= 0) & (preds <= 1)):
            probs = preds
        else:
            probs = preds.sigmoid()
        
        # Upsample predictions to target size
        preds_up = F.interpolate(
            probs, size=(h, w), mode="bilinear", align_corners=False
        )[..., :h, :w]
        
        preds_up = (preds_up.detach() > 0.5).squeeze(0).float()
        
        return preds_up


