"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv



class GraphConvolution(nn.Module):
    """
    GCN 
    """

    def __init__(self, in_dim, hidden_dim, dropout, act=None):
        super(GraphConvolution, self).__init__()
        self.conv = GCNConv(in_dim, hidden_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    

class GraphAttention(nn.Module):
    """
    GAT layer wrapper.
    """

    def __init__(self, in_dim, hidden_dim, dropout, act=None, num_heads=1, concat=True, v2=False):
        super(GraphAttention, self).__init__()
        self.concat = concat
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim


        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=concat,
            dropout=dropout  # attention dropout
        )
        self.dropout = dropout
        self.act = act

    def out_dim(self):
        return self.hidden_dim * self.num_heads if self.concat else self.hidden_dim

    def forward(self, x, edge_index, edge_attr):

        x = self.conv(x, edge_index, edge_attr)

        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x




class MLP_layer(nn.Module):
    """
    MLP layer for node classification.
    """

    def __init__(self, in_dim, hidden_dim, dropout, act=None):
        super(MLP_layer, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x



class HyboEuclideanConvolution(nn.Module):
    """
    Lorentz graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):
        super(HyboEuclideanConvolution, self).__init__()
        self.linear = HyboEuclideanLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = HyboEuclideanAgg(manifold, out_features, dropout, use_att)

    def forward(self, input):
        x, edge_index, edge_attr = input
        adj = torch.sparse_coo_tensor(edge_index, edge_attr, (x.size(0), x.size(0)))
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h, adj
        return output



class HyboEuclideanLinear(nn.Module):
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
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.nonlin = nonlin
        self.use_bias = bias

        self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)

        x = self.weight(self.dropout(x))
     
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight.weight, -stdv, stdv)

        if self.use_bias:
            nn.init.constant_(self.weight.bias, 0)



class HyboEuclideanAgg(nn.Module):
    """
    Euclidean aggregation layer based on Eq(5) from 'Fully Hyperbolic Neural Networks' (Chen, 2022)
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(HyboEuclideanAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = nn.Linear(in_features, in_features, bias=True)
            self.query_linear = nn.Linear(in_features, in_features, bias=True)
            self.bias = nn.Parameter(torch.zeros(()))
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = - self.manifold.sqdist(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
            denom = att_adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [N,1]
            output = support_t / denom

        else:
            support_t = torch.spmm(adj, x)  # sum aggregation

            # degree-normalize (mean aggregation)
            deg = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1).clamp_min(1e-8)
            output = support_t / deg

        return output
    