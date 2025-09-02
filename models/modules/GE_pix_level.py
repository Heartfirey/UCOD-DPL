import torch
from torch import nn
import torch.nn.functional as F


class GatedEnsembler(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.fuser = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=1),  
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1), 
        )

    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        _, _, h, w = l2.shape
        l1 = F.interpolate(l1, size=(h, w), mode='bilinear')
        l1_probs = torch.sigmoid(l1)
        fg_ratio_g = l1_probs.mean(dim=(1,2,3), keepdim=True)  # [B,1,1,1]
        fg_ratio_l = F.avg_pool2d(l1_probs.float(), 19, padding=19//2, stride=1)
        en_local = -fg_ratio_l * torch.log(fg_ratio_l.clamp(1e-5))
        en_local = 1 - en_local/en_local.max()
        l1_weight = (en_local + fg_ratio_g)/2
        y = l1 * l1_weight + l2 * (1 - l1_weight)
        return self.fuser(y), l1_weight