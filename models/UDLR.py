from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch import nn
from models.modules.ASR import EntropySelector
from models.modules.HRE import HRE
from models.modules.GE_pix_level import GatedEnsembler

class SparseRefiner(nn.Module):
    def __init__(
        self,
        config,
        window_size: int,
        threshold: float,
        dim: int=768,
    ) -> None:
        super().__init__()
        self.config = config
        self.selector = EntropySelector(threshold, window_size)
        self.HRE = HRE(window_size, dim)
        self.GE = GatedEnsembler(1)
        self.window_size = window_size
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def binary_iou(self, preds: torch.Tensor, targets: torch.Tensor, threshold=0.5) -> torch.Tensor:
        if preds.ndim == 4:
            preds = preds.squeeze(1)
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        if preds.dtype.is_floating_point:
            preds = (torch.sigmoid(preds) if preds.max() > 1 else preds)
            preds = (preds > threshold).int()

        targets = targets.int()

        intersection = (preds & targets).sum(dim=(1, 2)).float()
        union = (preds | targets).sum(dim=(1, 2)).float()
        
        iou = intersection / (union + 1e-6)
        return iou

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            config,
            config.window_size,
            config.threshold,
        )

    def cal_ex_loss(self, opt):
        loss = 0
        if not self.training:
            return loss, opt
        with torch.no_grad():
            l_preds = opt['preds']
            window_preds = opt['window_preds']
            mask = opt['mask']
            if mask.sum() == 0:
                return loss, opt
            h_preds = opt['h_targets']
            num = [x.sum() for x in mask]
            _, _, h, w = window_preds.shape
            l_preds = F.interpolate(l_preds, size=(h*self.window_size,w*self.window_size), mode='bilinear').sigmoid()>0.5
            l_preds = torch.nn.functional.unfold(l_preds.float(), kernel_size=(h, w), stride=(h, w)).reshape(-1, 1, h, w, self.window_size**2).permute(0,4,1,2,3).flatten(0,1)
            l_preds = l_preds[mask.flatten()]
            
            h_preds = h_preds[mask.flatten()]
            opt['window_targets'] = h_preds
            ious = self.binary_iou(h_preds, l_preds).view(-1,1,1,1)*1.5
            ious = ious.clamp(0,1)
        N = window_preds.shape[0]
        loss = ((ious * self.criterion(window_preds, h_preds) + (1-ious) * self.criterion(window_preds, l_preds)).mean())/2
        return loss, opt

    def forward(self, input_features, h_inputs, preds, h_targets=None):
        with torch.no_grad():
            l_input_features, h_inputs_features, mask, coords_list, entropy = self.selector(input_features, h_inputs, preds)
        h_preds, window_preds = self.HRE(l_input_features, h_inputs_features, mask, coords_list)
        outputs, GE_w = self.GE(preds, h_preds)
        
        opt={'mask':mask, 'entropy':entropy, 'h_preds':h_preds, 'window_preds':window_preds, 
             'GE_w':GE_w, 'preds':preds, 'coords_list':coords_list, 'h_targets':h_targets}
        ex_loss, opt = self.cal_ex_loss(opt)
        return  outputs, ex_loss, opt