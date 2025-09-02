import torch
import torch.nn as nn
import torch.nn.functional as F
        
class RevDecoder(nn.Module):
    def __init__(self, cfg, ema: bool = False):
        super().__init__()
        feature_dim = cfg.dim
        embedding_dim = 64
        
        self.embed_dim = embedding_dim
        
        self.decoupling = nn.Conv2d(feature_dim, 2 * embedding_dim, kernel_size=(1, 1))
        
        self.learnable_embedding = nn.Parameter(torch.randn(2, embedding_dim))
        
        self.conv_out_fg = nn.Conv2d(embedding_dim, 1, kernel_size=(1,1))
        self.conv_out_bg = nn.Conv2d(embedding_dim, 1, kernel_size=(1,1))
        
        self.ema = ema
        if ema:
            for param in self.parameters():
                param.detach_()

    def calc_orthogonal_loss(self, feature_1, feature_2, weight=1.0):
        dot_product = torch.bmm(feature_1, feature_2.transpose(1, 2))
        identity = torch.eye(feature_1.size(1)).to(feature_1.device)
        loss = ((dot_product * (1 - identity)).pow(2)).mean()
        return weight * loss
        
    def forward(self, x, get_bg_mask=False):
        if type(x) is list: x = x[-1]
        
        B, _, H, W = x.shape
        decoupled_feat = self.decoupling(x)
        dfeatures_1, dfeatures_2 = torch.chunk(decoupled_feat, 2, dim=1)
        
        features_1 = dfeatures_1.view(B, self.embed_dim, -1).permute(0, 2, 1)
        features_2 = dfeatures_2.view(B, self.embed_dim, -1).permute(0, 2, 1)
        features_1 = F.normalize((features_1 * self.learnable_embedding[0]), p=2, dim=1)
        features_2 = F.normalize((features_2 * self.learnable_embedding[1]), p=2, dim=1)
        if not self.ema:
            extra_loss = self.calc_orthogonal_loss(features_1, features_2)
        
        features_1 = features_1.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        features_2 = features_2.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        
        attention_1 = F.sigmoid(features_1 * dfeatures_1) + dfeatures_1
        attention_2 = F.sigmoid(features_2 * dfeatures_2) + dfeatures_2
        
        fg_mask = self.conv_out_fg(attention_1)
        bg_mask = self.conv_out_bg(attention_2)
        
        if not self.ema:
            return fg_mask, bg_mask, extra_loss
        elif get_bg_mask:
            return fg_mask, bg_mask
        else:
            return fg_mask
