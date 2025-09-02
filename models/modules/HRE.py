from typing import Any, Dict
import torch
from torch import nn
import torch.nn.functional as F
from models.modules.CSF import CSF

class HRE(nn.Module):
    def __init__(
        self,
        window_size,
        dim=768
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.CSF = CSF()


    def concate_windows(self, windows, positions, candidate_windows_mask):
        N, C, H, W = windows.shape
        B = candidate_windows_mask.shape[0]
        device = windows.device
        num = [batch.sum() for batch in candidate_windows_mask]

        full_mask = torch.zeros((B, C, H*self.window_size, W*self.window_size), device=device)
        counter = torch.zeros((B, 1, H*self.window_size, W*self.window_size), device=device)
        last_num=0
        for b in range(B):
            if num[b]==0:
                continue
            now_num = num[b] + last_num
            y = positions[last_num:now_num, 0] * H
            x = positions[last_num:now_num, 1]* W
            for i in range(last_num,now_num):
                full_mask[b, :, y[i-last_num]:y[i-last_num]+H, x[i-last_num]:x[i-last_num]+W] += windows[i, :, :, :]
                counter[b, :, y[i-last_num]:y[i-last_num]+H, x[i-last_num]:x[i-last_num]+W] += 1.0
            last_num = now_num

        full_mask = full_mask / (counter + 1e-6)
        return full_mask

    def forward(self, l_input_features, h_inputs_features, candidate_windows_mask, coords_list):
        preds = self.CSF(l_input_features, h_inputs_features)
        full_mask = self.concate_windows(preds, coords_list, candidate_windows_mask)
        return full_mask, preds

