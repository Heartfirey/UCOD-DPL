import torch
from torch import nn
from torch.nn import functional as F
from models.uscod import baseline
from models.modules.ocm import ocm
from data.utils.feature_extractor import build_feature_extractor
from engine.config.config import CfgNode
from safetensors.torch import load_file
from peft import get_peft_model, LoraConfig, TaskType
import os 
import copy
from engine.utils.logger import simple_logger

logger = simple_logger("INFO")

class ViTLoraWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ViT = model

    def forward(self, **kwargs):
        pixel_values = kwargs.get("pixel_values")
        return self.ViT(pixel_values)

def get_full_model(cfg=None, checkpoint_path=None):
    if cfg == None:
        cfg = load_config()
    model = baseline(cfg.model_cfg)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(load_file(checkpoint_path, device="cuda"))
    feature_extractor = build_feature_extractor(cfg.dataset_cfg.feature_extractor_cfg)
    feature_extractor = ViTLoraWrapper(feature_extractor)
    # freeze_model(feature_extractor)
    feature_extractor = load_lora(cfg.lora_cfg, feature_extractor)
    return full_model(cfg, feature_extractor, model)

def freeze_model(model, exclude_keywords=None):
    if exclude_keywords is None:
        exclude_keywords = []

    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in exclude_keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False

def load_lora(config: CfgNode, model):
    r = getattr(config, 'r', 2)
    if r == 0:
        return model
    lora_alpha = getattr(config, 'lora_alpha', 4)
    lora_dropout = getattr(config, 'lora_dropout', 0.05)
    bias = getattr(config, 'bias', 'none')
    target_modules = getattr(config, 'target_modules', ['query', 'value', 'key'])

    task_type_str = getattr(config, 'lora_task_type', 'FEATURE_EXTRACTION')
    try:
        task_type = TaskType[task_type_str]
    except KeyError:
        raise ValueError(f"Unsupported task_type: {task_type_str}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=task_type
    )

    model = get_peft_model(model, lora_config)
    return model

def load_config(config_path='configs/uscod/Look_Twice_.py'):
    cfg = CfgNode.load_with_base(config_path)
    cfg = CfgNode(cfg)
    return cfg

class full_model(nn.Module):
    def __init__(self, config, backbone, decoder):
        super().__init__() 
        self.config = config
        self.enable_ocm = config.model_cfg.enable_ocm
        self.backbone = backbone
        self.backbone_ema = copy.deepcopy(backbone)
        if self.enable_ocm:
            self.ocm = ocm(config.model_cfg.ocm_cfg)
        self.freeze_model(self.backbone_ema)
        self.decoder = decoder
        if config.model_cfg.freeze_lora:
            logger.warning('The lora layer has been freezed!')
            self.freeze_lora()
        self.key = None
        
        self.backbone.base_model.model.ViT.encoder.layer[-1].attention.attention.key.register_forward_hook(self.hook_fn_key)
        self.backbone_ema.base_model.model.ViT.encoder.layer[-1].attention.attention.key.register_forward_hook(self.hook_fn_key)
    def hook_fn_key(self, module, input, output):
        output = output[:,1:,:]
        b, l, c = output.shape
        h = w = int(l** 0.5)
        output = output.reshape(b, h, w, c)
        output = output.permute(0,3,1,2)
        #TODO
        ih = iw = 68
        output = F.interpolate(output, size=(ih,iw), mode='bilinear')
        self.key = output

    def forward(self, inputs, ema=False, get_hidden_feature=False):
        if ema:
            self.backbone_ema(pixel_values=inputs)
        else:
            outputs = self.backbone(pixel_values=inputs)
        if self.enable_ocm and not ema:
            self.ocm(list(outputs.attentions))
            loss_ocm = self.ocm.get_loss()
        if get_hidden_feature:
            return self.key
        
        if not ema:
            preds, preds_rev, extra_loss = self.decoder(self.key, ema=ema)
            if self.enable_ocm:
                extra_loss += loss_ocm
            return preds, preds_rev, extra_loss
        else:
            preds = self.decoder(self.key, ema=ema)
            return preds
    
    def freeze_model(self, model, exclude_keywords=None):
        if exclude_keywords is None:
            exclude_keywords = []

        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in exclude_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_lora(self):
        for name, param in self.backbone.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
    def active_lora(self):
        for name, param in self.backbone.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def load_state_dict(self, state_dict):
        self.decoder.load_state_dict(state_dict)
