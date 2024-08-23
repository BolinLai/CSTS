import torch
import torch.nn.functional as F


def frame_softmax(logits, temperature):
    batch_size, T = logits.shape[0], logits.shape[2]
    H, W = logits.shape[3], logits.shape[4]
    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, -1, T, H * W)
    atten_map = F.softmax(logits / temperature, dim=-1)
    atten_map = atten_map.view(batch_size, -1, T, H, W)
    return atten_map


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    Copied from https://github.com/showlab/EgoVLP/blob/main/model/model.py
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
