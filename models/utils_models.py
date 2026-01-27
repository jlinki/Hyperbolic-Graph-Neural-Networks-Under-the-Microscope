"""Utility functions for models."""
from torch import nn as nn

import manifolds
from manifolds.stmath import artanh, tanh
import torch

def make_manifold(manifold_name, c, learnable=False):
    if manifold_name == 'PoincareBall':
        return manifolds.PoincareBall(c=c, learnable=learnable)
    elif manifold_name == 'Lorentz':
        return manifolds.Lorentz(k=c, learnable=learnable)
    elif manifold_name == 'Euclidean':
        return manifolds.Euclidean()
    else:
        raise ValueError(f"Unknown manifold name: {manifold_name}")
    

### Utils for Chami Layers ###

def mobius_matvec(manifold, m, x, c):
    """Performs the Möbius matrix-vector multiplication."""
    if manifold.name.lower() == "lorentz":
        x_0 = manifold.logmap0(x)
        mx_0 = x_0 @ m.transpose(-1, -2)
        mx = manifold.expmap0(mx_0)
        res = manifold.projx(mx)
        return res
    if manifold.name.lower() == "poincareball":
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-15)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-15)
        sqrt_c = c ** 0.5
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_0, res_c)
        res = manifold.projx(res)
        return res
    
def proj_tan0(manifold, x):
    if manifold.name.lower() == "lorentz":
        narrowed = x.narrow(-1, 0, 1)
        vals = torch.zeros_like(x)
        vals[:, 0:1] = narrowed
        return x - vals
    elif manifold.name.lower() == "poincareball":
        return x
    

def mobius_add(manifold, x, y, c, dim=-1):
    """Performs the Möbius addition."""
    if manifold.name.lower() == "lorentz":
        u = manifold.logmap0(y)
        v = manifold.transp0(x, u)
        res = manifold.expmap(v, x)
        res = manifold.projx(res)
        return res
    if manifold.name.lower() == "poincareball":
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = x.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        res = num / denom.clamp_min(1e-15)
        res = manifold.projx(res)
        return res


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, scores):
        probs = 1. / (torch.exp(((-scores - self.r) / self.t).clamp_max(20)) + 1.0)
        return probs


class MarginLoss(torch.nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, preds, data_edge_label=None):
        pos_scores = preds[data_edge_label==1.0]
        neg_scores = preds[data_edge_label==0.0]
        return torch.nn.functional.relu(self.margin - pos_scores[:, None] + neg_scores[None, :]).mean()


class StressLoss(nn.Module):
    """
    Stress loss for pairwise distances (1D tensors).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D_pred: torch.Tensor, D_true: torch.Tensor) -> torch.Tensor:
        # D_pred, D_true: shape [num_pairs]
        weights = 1.0 / (D_true ** 2 + self.eps)
        loss = (weights * (D_pred - D_true) ** 2).mean()
        return loss


class StressLossNR(nn.Module):
    """
    Stress loss for pairwise distances (1D tensors) ignoring the target 0 node.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D_pred: torch.Tensor, D_true: torch.Tensor) -> torch.Tensor:
        # D_pred, D_true: shape [num_pairs]
        mask = torch.where(D_true > 0)[0]
        weights = 1.0 / (D_true ** 2 + self.eps)
        loss = (weights[mask] * (D_pred[mask] - D_true[mask]) ** 2).mean()
        return loss


class BCELoss(torch.nn.Module):
    def __init__(self, decoder) -> None:
        super().__init__()
        self.dc_function = decoder
        self.loss = torch.nn.BCELoss()
    def forward(self, preds, data_edge_label=None):
        return self.loss(self.dc_function(preds), data_edge_label)
