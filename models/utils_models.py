"""Utility functions for models."""

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
    

