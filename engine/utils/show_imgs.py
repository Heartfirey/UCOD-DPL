import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.patches as patches

def draw_bboxes_on_image_and_save(image_input, bbox_list, save_path, mask_tensor=None):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input 必须是图像路径 (str) 或 PIL.Image 对象")

    W, H = image.size

    # 设置绘图
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 如果提供了 mask_tensor，进行处理和叠加
    if mask_tensor is not None:
        if not (isinstance(mask_tensor, torch.Tensor) and mask_tensor.dim() == 3 and mask_tensor.size(0) == 1):
            raise ValueError("mask_tensor 必须是形状为 [1, H, W] 的 torch.Tensor")
        
        # 调整 mask 大小并转成 PIL 图像
        resized_mask = TF.resize(mask_tensor, [H, W])  # [1, H, W]
        mask_pil = TF.to_pil_image(resized_mask.squeeze(0))

        # 叠加 mask
        ax.imshow(mask_pil, cmap='jet', alpha=0.5, interpolation='none')

    # 绘制所有 bbox
    for bbox in bbox_list:
        x, y, w, h = bbox
        x_pixel = x * W
        y_pixel = y * H
        w_pixel = w * W
        h_pixel = h * H

        rect = patches.Rectangle(
            (x_pixel, y_pixel), w_pixel, h_pixel,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()