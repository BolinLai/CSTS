#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from slowfast.utils.utils import frame_softmax


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = - (5 * y * F.logsigmoid(x) + (1 - y) * torch.log(1 - torch.sigmoid(x)))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class KLDiv(nn.Module):
    """
      KL divergence for 3D attention maps
    """
    def __init__(self):
        super(KLDiv, self).__init__()
        self.register_buffer('norm_scalar', torch.tensor(1, dtype=torch.float32))

    def forward(self, pred, target=None):
        # get output shape
        batch_size, T = pred.shape[0], pred.shape[2]
        H, W = pred.shape[3], pred.shape[4]

        # N T HW
        atten_map = pred.view(batch_size, T, -1)
        log_atten_map = torch.log(atten_map + 1e-10)

        if target is None:
            # uniform prior: this is really just neg entropy
            # we keep the loss scale the same here
            log_q = torch.log(self.norm_scalar / float(H * W))
            # \sum p logp - log(1/hw) -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - log_q
        else:
            log_q = torch.log(target.view(batch_size, T, -1) + 1e-10)
            # \sum p logp - \sum p logq -> N T
            kl_losses = (atten_map * log_atten_map).sum(dim=-1) - (atten_map * log_q).sum(dim=-1)
        # N T -> N
        norm_scalar = T * torch.log(self.norm_scalar * H * W)
        kl_losses = kl_losses.sum(dim=-1) / norm_scalar
        kl_loss = kl_losses.mean()
        return kl_loss


class FLoss(nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        # weights_1 = self.build_weight_from_target(target)
        # weights_1 = torch.from_numpy(weights_1).to(target.device)
        weights = self.build_weight_from_target_pytorch(target)
        loss = nn.functional.binary_cross_entropy(input, target, weight=weights)
        return loss

    def build_weight_from_target(self, target):
        target = target.data.cpu().numpy()
        batch_num = target.shape[0]
        frame_num = target.shape[1]
        image_width = target.shape[-1]
        weightmat = np.empty_like(target)
        weightmat.astype(float)

        for bb in range(batch_num):
            for tt in range(frame_num):
                target_im = target[bb, tt, :, :].squeeze()
                x, y = np.where(target_im == np.amax(target_im))
                x = x.mean()
                y = y.mean()
                a = np.arange(image_width)
                b = np.arange(image_width)
                a = a - x
                b = b - y
                a = np.tile(a, (image_width, 1))
                b = np.tile(b, (image_width, 1))
                a = np.transpose(a)
                dist = a**2 + b**2
                dist = (np.sqrt(dist) + 1) / image_width
                dist = np.reciprocal(dist)
                weightmat[bb, tt, :, :] = dist
        return weightmat


    def build_weight_from_target_pytorch(self, target):
        batch_num = target.size(0)
        frame_num = target.size(1)
        image_width = target.size(-1)
        weightmat = torch.zeros_like(target, device=target.device)

        for bb in range(batch_num):
            for tt in range(frame_num):
                target_im = target[bb, tt, :, :]
                x, y = torch.where(target_im == torch.max(target_im))
                x = x.float().mean()
                y = y.float().mean()
                a = torch.arange(image_width, device=target.device)
                b = torch.arange(image_width, device=target.device)
                a = a - x
                b = b - y
                a = a.repeat(image_width, 1)
                b = b.repeat(image_width, 1)
                a = torch.transpose(a, 0, 1)
                dist = a ** 2 + b ** 2
                dist = (torch.sqrt(dist) + 1) / image_width
                dist = torch.reciprocal(dist)
                weightmat[bb, tt, :, :] = dist

        return weightmat


class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        mask = torch.eye(x.shape[0]).cuda()

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1))
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j


class KLDiv_plus_FLoss(nn.Module):
    def __init__(self, alpha=1):
        super(KLDiv_plus_FLoss, self).__init__()
        self.alpha = alpha
        self.KLDiv = KLDiv()
        self.FLoss = FLoss()

    def forward(self, pred, target=None):
        kldiv = self.KLDiv(frame_softmax(pred, temperature=2), target)
        floss = self.FLoss(torch.sigmoid(pred), target)
        return kldiv + self.alpha * floss



_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "kldiv": KLDiv,
    "floss": FLoss,
    "egonce": EgoNCE,
    "kldiv+floss": KLDiv_plus_FLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
