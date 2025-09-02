import torch
from torch import nn
from torch.nn import functional as F
from models.modules.DBA import RevDecoder

from engine.config.config import CfgNode


class baseline(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__() 
        self.cfg = cfg
        self.decoder = RevDecoder(cfg)
        self.decoder_ema = RevDecoder(cfg, ema=True)
            
    def forward(self, batched_inputs, ema: bool=False):
        if ema:
            with torch.no_grad():
                preds = self.decoder_ema(batched_inputs)
        else:
            preds = self.decoder(batched_inputs)
        return preds