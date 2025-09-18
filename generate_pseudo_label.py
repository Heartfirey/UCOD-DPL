import argparse
import os
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import AutoModel
from torchvision import transforms
from tqdm import tqdm

from data.utils.found_bkg_mask import compute_img_bkg_seg
from engine.utils.fileio.backend import MetaListPickleIO

import torch.nn.functional as F

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


key = None
model = None
transform = None


def hook_fn_key(module, input, output):
    """Hook function to capture key features."""
    global key
    key = output.detach()


def refine_post_process(mask, area_threshold=4):
    """Refine mask by removing small isolated components."""
    mask_np = mask.numpy().astype(np.uint8).squeeze()
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    refined_mask = mask_np.copy()
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area < area_threshold:
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            
            component_mask = (labels_im[y:y+height, x:x+width] == label)

            x_start = max(x - 1, 0)
            y_start = max(y - 1, 0)
            x_end = min(x + width + 1, mask_np.shape[1])
            y_end = min(y + height + 1, mask_np.shape[0])
            
            surrounding = refined_mask[y_start:y_end, x_start:x_end].copy()
            comp_y, comp_x = np.where(component_mask)
            comp_y += y - y_start
            comp_x += x - x_start
            surrounding_mask = np.ones_like(surrounding, dtype=bool)
            surrounding_mask[comp_y, comp_x] = False
            surrounding_pixels = surrounding[surrounding_mask]

            component_label = refined_mask[y + height // 2, x + width // 2]
            opposite_label = 1 - component_label
            
            if np.all(surrounding_pixels == opposite_label):
                refined_mask[y:y+height, x:x+width][component_mask] = opposite_label
                
    return torch.tensor(refined_mask).unsqueeze(0).float()


def generate_mask(image_path, th_bkg=0.6):
    """Generate pseudo label mask for a single image."""
    image = Image.open(image_path).convert('RGB')
    inputs = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(inputs)
        attn = outputs.attentions[-1]
        
        b, N, c = key.shape
        h_f = w_f = int((N-1)**0.5)
        nh = attn.shape[1]
        
        mask, _ = compute_img_bkg_seg(
            attentions=attn, 
            feats=key, 
            featmap_dims=(h_f, w_f), 
            th_bkg=th_bkg, 
            dim=c//nh
        )
        
        mask = (1-mask).to('cpu')
        mask = refine_post_process(mask)
        
    return mask


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate pseudo labels for COD datasets using DINOv2")
    parser.add_argument('--dataset', type=str, default='TR-CAMO+TR-COD10K', 
                       help='Dataset name (can use + to combine multiple datasets)')
    parser.add_argument('--image_path', type=str, default='/data/ywq/dataset/COD/{}/im',
                       help='Template path for images (use {} as placeholder for dataset name)')
    parser.add_argument('--cache_path', type=str, default='./datasets/cache/pseudo_label_cache/',
                       help='Path to save cache files')
    
    args = parser.parse_args()
    
    # Initialize global variables
    global model, transform
    model = AutoModel.from_pretrained('facebook/dinov2-base', output_attentions=True)
    model.encoder.layer[-1].attention.attention.key.register_forward_hook(hook_fn_key)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_path, exist_ok=True)
    
    # Initialize cache
    cacheio = MetaListPickleIO(base_path=os.path.join(args.cache_path, args.dataset))
    
    # Collect image paths
    image_paths = []
    for dataset in args.dataset.split('+'):
        dir_path = args.image_path.format(dataset)
        if not os.path.exists(dir_path):
            raise ValueError(f'Image path {dir_path} does not exist!')
        
        for name in os.listdir(dir_path):
            image_paths.append(os.path.join(dir_path, name))
    
    image_paths = sorted(image_paths)
    print(f'Found {len(image_paths)} images from {args.dataset}.')
    
    # Generate masks
    mask_list = []
    for file_path in tqdm(image_paths, desc="Generating pseudo labels"):
        try:
            mask = generate_mask(file_path, 0.6)
            mask_list.append(mask)

            upsampled_tensor = F.interpolate(mask.unsqueeze(0), size=(68,68), mode='nearest').squeeze(0)
            output_image_data = upsampled_tensor.squeeze().numpy() # 移除维度，得到 (68, 68)
            output_image_data = (output_image_data * 255).astype(np.uint8) # 转换为 0-255 的整数类型
            img = Image.fromarray(output_image_data)
            output_dir = "/data/ywq/USCOD_newest/datasets/cache/visulization"
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, os.path.basename(file_path))
            img.save(output_filepath)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Save to cache
    cacheio.dump_list(mask_list)
    print(f"Successfully generated {len(mask_list)} pseudo labels and saved to cache.")


if __name__ == "__main__":
    main()