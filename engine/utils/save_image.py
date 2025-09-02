import torch
import os
import numpy as np
from PIL import Image

def save_tensor_binary_mask_as_image(binary_mask: torch.Tensor, save_path: str) -> None:
    """
    Save a binary mask tensor as an image file.

    For batched tensors (batch size > 1), saves each mask as a separate PNG file
    in a directory named after the save_path (without extension).
    For single masks, saves as a single PNG file.

    Args:
        binary_mask: Binary mask tensor to save
        save_path: Path where to save the image(s)
    """
    try:
        if binary_mask.dim() == 4 and binary_mask.shape[0] > 1:
            _save_batch_masks(binary_mask, save_path)
        else:
            _save_single_mask(binary_mask, save_path)
    except Exception as e:
        print(f"Error saving mask to {save_path}: {e}")


def _save_batch_masks(binary_mask: torch.Tensor, save_path: str) -> None:
    """
    Save multiple masks from a batch tensor.
    
    Args:
        binary_mask: Batch of binary masks
        save_path: Base path for saving
    """
    batch_size = binary_mask.shape[0]
    folder_path = os.path.splitext(save_path)[0]
    os.makedirs(folder_path, exist_ok=True)
    
    for i in range(batch_size):
        mask_slice = binary_mask[i].squeeze()
        if mask_slice.dim() != 2:
            print(f"Warning: Unexpected mask dimensions after squeeze: {mask_slice.shape} for batch item {i}")
            continue
        
        mask_np = (mask_slice.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(mask_np, mode='L')
        img_path = os.path.join(folder_path, f"{i}.png")
        img.save(img_path)


def _save_single_mask(binary_mask: torch.Tensor, save_path: str) -> None:
    """
    Save a single mask tensor.
    
    Args:
        binary_mask: Single binary mask tensor
        save_path: Path where to save the image
    """
    mask_2d = binary_mask.squeeze()
    
    if mask_2d.dim() != 2:
        print(f"Warning: Could not save mask due to unexpected shape: {binary_mask.shape}")
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy and save
    binary_mask_np = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(binary_mask_np, mode='L')
    
    # Ensure PNG extension
    save_path = save_path.replace('.jpg', '.png')
    img.save(save_path)