from typing import Any, Dict
import torch
from torch import nn
import torch.nn.functional as F
from models.modules.mlp import CrossAttentionBlock, Attention, Block

class CSF(nn.Module):
    def __init__(
        self,
        dim=768
    ) -> None:
        super().__init__()
        self.attn = CrossAttentionBlock(dim=dim)
        self.depthwise_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,  
            kernel_size=7,
            padding=3,
            groups=dim  
        )
        self.mask_dec = nn.Conv2d(
            in_channels=dim,
            out_channels=1,  
            kernel_size=1,
            padding=0
        )

    def _BCHW_to_BLC(self, input):
        input = input.flatten(2,3).permute(0,2,1)
        return input

    def _BLC_to_BCHW(self, input):
        B, L, C = input.shape
        h = w = int(L**0.5)
        input = input.reshape(B, h, w, C).permute(0,3,1,2)
        return input

    def forward(self, l_inputs, h_inputs):
        outputs = self.attn(query=self._BCHW_to_BLC(h_inputs), context=self._BCHW_to_BLC(l_inputs))
        outputs = self._BLC_to_BCHW(outputs)
        outputs = self.depthwise_conv(outputs)
        preds_mask = self.mask_dec(outputs)
        return preds_mask