"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_scatter import scatter

from layers.att_layers import DenseAtt, SparseAtt
from models.utils_models import mobius_add, mobius_matvec, proj_tan0



class LorentzGraphConvolution(nn.Module):
    """
    Lorentz graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att)

    def forward(self, input):
        x, edge_index, edge_attr = input
        adj = torch.sparse_coo_tensor(edge_index, edge_attr, (x.size(0), x.size(0)))
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h, adj
        return output



class LorentzLinear(nn.Module):
    """
    Lorentz Linear Layer based on Eq(3) from 'Fully Hyperbolic Neural Networks' (Chen, 2022).
    Learns embeddings vectors on the hyperboloid using algebra - no expmap.
    """
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.nonlin = nonlin
        self.use_bias = bias

        self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

        self.reset_parameters()

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)

        x = self.weight(self.dropout(x))

        if self.manifold.name.lower() == "lorentz":
            # Separate time and spatial components
            x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
            time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1

            # Compute scaling factor for spatial part
            scale = (time * time - 1) / \
                (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Return Lorentz vector
            x_lorentz = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
            return x_lorentz
        
        else:      
            return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight.weight, -stdv, stdv)

        # Zero out the first column of weight (time-like component) - the parameters are initialized in the tangent space of the origin 
        with torch.no_grad():
            self.weight.weight[:, 0] = 0

        if self.use_bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer based on Eq(5) from 'Fully Hyperbolic Neural Networks' (Chen, 2022)
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            support_t = torch.spmm(adj, x)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic Graph Convolution Layer.
    """

    def __init__(self, manifold_in, manifold_out, in_features, out_features, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()

        self.manifold_in = manifold_in
        self.manifold_out = manifold_out

        self.linear = HypLinearChami(self.manifold_in, in_features, out_features, dropout, use_bias)
        self.agg = HypAggChami(self.manifold_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypActChami(self.manifold_in, self.manifold_out, act)


    def forward(self, input):
        x, edge_index, edge_attr = input
        adj = torch.sparse_coo_tensor(edge_index, edge_attr, (x.size(0), x.size(0)))
        h = self.linear(x)
        h = self.agg(h, adj)
        h = self.hyp_act(h)

        output = h, adj
        return output


class HypLinearChami(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout, use_bias):
        super(HypLinearChami, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)


    def forward(self, x):
        # # Conversion from Chami curvature parameter to Geoopt-style curvature
        if self.manifold.name.lower() == "lorentz":
            c = 1.0 / self.manifold.k
        elif self.manifold.name.lower() == "poincareball":
            c = self.manifold.c

        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = mobius_matvec(self.manifold, drop_weight, x, c)


        if self.use_bias:
            # project bias to tangent space at origin
            bias = proj_tan0(self.manifold, self.bias.view(1, -1))
            # project to manifold
            hyp_bias = self.manifold.expmap0(bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            #parallel transport
            res = mobius_add(self.manifold, res, hyp_bias, c)

        return res



class HypAggChami(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(HypAggChami, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        self.sparse_att = True
        if self.use_att:
            if self.sparse_att:
                self.att = SparseAtt(in_features, dropout)
            else:
                self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x)
        if self.use_att:
            if self.local_agg:
                if self.sparse_att:
                    adj_att, x_i, x_j, row, _= self.att(x_tangent, adj)
                    v_ij = adj_att.unsqueeze(-1) * self.manifold.logmap(x_i, x_j) # shape [E, d] 
                    support_t = scatter(v_ij, row, dim=0, reduce='sum') # shape [n, d]
                else:
                    x_local_tangent = []
                    for i in range(x.size(0)):
                        x_local_tangent.append(self.manifold.logmap(x[i], x))
                    x_local_tangent = torch.stack(x_local_tangent, dim=0) # shape [n, n, d]
                    adj_att = self.att(x_tangent, adj) # shape [n, n]
                    support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1) # shape [n, d]

                output = self.manifold.projx(self.manifold.expmap(x, support_t))
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.projx(self.manifold.expmap0(support_t))
        return output


class HypActChami(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypActChami, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        if self.act is not None:
            xt = self.act(self.manifold_in.logmap0(x))
        else:
            xt = self.manifold_in.logmap0(x)
        xt = proj_tan0(self.manifold_out, xt)

        return self.manifold_out.projx(self.manifold_out.expmap0(xt))