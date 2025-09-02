import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
import timm
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from transformers import pipeline

from engine.config import CfgNode
from engine.utils.logger import simple_logger

logger = simple_logger("INFO")

def build_feature_extractor(config: CfgNode):
    assert config.backbone_type == 'huggingface'
    bb_weight = Path(config.backbone_weights).expanduser()
    if 'dino' in config.type:
        try:
            model = transformers.AutoModel.from_pretrained(bb_weight, output_attentions=True)
        except Exception as e:
            logger.warning('Failed to load model from local weights, trying to download from huggingface')
            os.makedirs(config.backbone_weight_base, exist_ok=True)
            cache_dir = Path(config.backbone_weight_base).expanduser()
            model = transformers.AutoModel.from_pretrained(config.backbone, output_attentions=True, cache_dir=cache_dir)
    else:
        logger.warning('{DINO}-type networks are not currently supported.'.format(config.type))
        raise ValueError(f"Unsupported model type: {config.type}")
    return model

class backbone(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = build_feature_extractor(config)
        self.key = None
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
        self.feature_extractor.encoder.layer[-1].attention.attention.key.register_forward_hook(self.hook_fn_key) 
        self.feature_extractor.eval()
        self.feature_extractor.cuda()
    
    def hook_fn_key(self, module, input, output):
        self.key = output.detach()

    def forward(self, input, reshape_keys = True):
        with torch.no_grad():
            if 'dinov2' in self.config.backbone:
                outputs = self.feature_extractor(input)
            else:
                outputs = self.feature_extractor(input, interpolate_pos_encoding=True)
        if reshape_keys:
            B, L, C = self.key.shape
            H = W = int((L-1)**0.5)
            self.key = self.key[:,1:,:].reshape(B, H, W, C).permute(0,3,1,2)
        return outputs, self.key   

