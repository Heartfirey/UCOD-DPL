import torch
import torch.nn.functional as F
from typing import Tuple
def compute_img_bkg_seg(
    attentions,
    feats,
    featmap_dims,
    th_bkg,
    up_size: int=None,
    dim=64,
    epsilon: float = 1e-10,
    apply_weights: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    inputs
       - attentions [B, ]
    """
    
    w_featmap, h_featmap = featmap_dims
    if up_size == None:
        up_size = w_featmap
    nb, nh, _ = attentions.shape[:3]
    # we keep only the output patch attention
    att = attentions[:, :, 0, 1:].reshape(nb, nh, -1)
    att = att.reshape(nb, nh, w_featmap, h_featmap)
    att = F.interpolate(att, size=(up_size,up_size), mode='bilinear')
    descs = feats[:,1:,]
    
    
    # -----------------------------------------------
    # Inspired by CroW sparsity channel weighting of each head CroW, Kalantidis etal.
    threshold = torch.mean(att.reshape(nb, -1), dim=1)  # Find threshold per image
    Q = torch.sum(
        att.reshape(nb, nh, up_size * up_size) > threshold[:, None, None], axis=2
    ) / (up_size * up_size)
    beta = torch.log(torch.sum(Q + epsilon, dim=1)[:, None] / (Q + epsilon))

    # Weight features based on attention sparsity
    
    if apply_weights:
        descs = (descs.reshape(nb, -1, nh, dim) * beta[:, None, :, None]).reshape(
            nb, -1, nh * dim
        )
    else:
        descs = (descs.reshape(nb, -1, nh, dim)).reshape(
            nb, -1, nh * dim
        )

    descs = descs.reshape(nb, w_featmap, h_featmap, -1)
    descs = descs.permute(0,3,1,2)
    descs = F.interpolate(descs, size=(up_size,up_size), mode='bilinear')
    descs = descs.permute(0,2,3,1)
    descs = descs.reshape(nb, -1, nh * dim)
    w_featmap = h_featmap = up_size
    # -----------------------------------------------
    # Compute cosine-similarities
    descs = F.normalize(descs, dim=-1, p=2)
    cos_sim = torch.bmm(descs, descs.permute(0, 2, 1))

    # -----------------------------------------------
    # Find pixel with least amount of attention
    if apply_weights:
        att = att.reshape(nb, nh, w_featmap, h_featmap) * beta[:, :, None, None]
    else:
        att = att.reshape(nb, nh, w_featmap, h_featmap) 
    id_pixel_ref = torch.argmin(torch.sum(att, axis=1).reshape(nb, -1), dim=-1)

    # -----------------------------------------------
    # Mask of definitely background pixels: 1 on the background
    cos_sim = cos_sim.reshape(nb, -1, w_featmap * h_featmap)

    bkg_mask = (
        cos_sim[torch.arange(cos_sim.size(0)), id_pixel_ref, :].reshape(
            nb, w_featmap, h_featmap
        )
        > th_bkg
    )  # mask to be used to remove background

    fn_mask = (1 - bkg_mask.float())

    sim_map = cos_sim[torch.arange(cos_sim.size(0)), id_pixel_ref, :].reshape(
            nb, w_featmap, h_featmap
        ).float()
    sim_map = 1 - sim_map
    sim_map = (sim_map / (sim_map.max() + 1e-10))
    return bkg_mask.float(), (sim_map * fn_mask).float()