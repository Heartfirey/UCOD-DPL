from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

class EntropySelector(nn.Module):
    def __init__(self, threshold: float, window_size: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.window_size = window_size

    def window_sets(self, mask, input_features, h_inputs):
        with torch.no_grad():
            h_inputs = h_inputs.flatten(0,1)
            num = [x.sum() for x in mask]
            mask = mask.flatten()
            h_input_features = h_inputs[mask]
            l_input_features = torch.repeat_interleave(input_features, torch.stack(num), dim=0)
            return l_input_features, h_input_features

    def get_position(self, mask):
        B = mask.shape[0]
        H = W = self.window_size
        device = mask.device

        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)

        coords = torch.stack([grid_y, grid_x], dim=-1).flatten(0,1)  # (H, W, 2)

        coords_list = []
        for b in range(B):
            mask_b = mask[b].flatten()          # (H, W)
            coords_b = coords[mask_b]       # (N_b, 2)
            coords_list.append(coords_b)    # append (N_b, 2)

        return torch.cat(coords_list, dim=0)

    def forward(self, input_features:torch.Tensor, h_inputs:torch.Tensor, preds) -> torch.Tensor:
        if torch.all((preds >= 0) & (preds <= 1)):
            probs = preds
        else:
            probs = preds.sigmoid()
        entropy = -probs * torch.log(probs.clamp(1e-5))
        scores = F.adaptive_avg_pool2d(entropy.float(), output_size=(self.window_size, self.window_size))
        mask = scores > self.threshold
        l_input_features, h_inputs_features = self.window_sets(mask, input_features, h_inputs)
        coords_list = self.get_position(mask)
        return l_input_features, h_inputs_features, mask, coords_list, entropy