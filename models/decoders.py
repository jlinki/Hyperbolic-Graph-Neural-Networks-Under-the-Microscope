"""Graph decoders."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import geoopt

from layers.layers import GraphConvolution, GraphAttention
from models.utils_models import make_manifold



class Decoder(nn.Module):
    """
    Abstract Decoder base class for graph tasks such as node classification and link prediction.
    The decoder takes node embeddings (and optionally the adjacency matrix)
    and produces outputs for downstream prediction tasks.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.cls = None # classifier to be set in the subclass

    def decode(self, x, edge_index=None, edge_attr=None):

        if self.cls is None:
            raise NotImplementedError("Decoder subclass must define self.cls.")
    
        def call_cls(cls,*inputs):
            output = cls(*inputs)
            if isinstance(output, tuple):
                return output[0]
            return output   

        try:
            return call_cls(self.cls, x, edge_index, edge_attr)
        except TypeError:
            return call_cls(self.cls, x)

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.decode(x, edge_index, edge_attr)


    

class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, args):
        super(GCNDecoder, self).__init__()
        self.cls = GCNConv(args.hidden_dim, args.n_classes)
        #self.cls = GraphConvolution(args.hidden_dim, args.n_classes, args.dropout)


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, args):
        super(GATDecoder, self).__init__()
        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None
        self.cls = GraphAttention(
            in_dim=args.hidden_dim*args.num_heads,
            hidden_dim=args.n_classes,
            dropout=args.dropout,
            act=self.act,
            num_heads=1,
            concat=True
        )

class LorentzDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, args):
        super(LorentzDecoder, self).__init__()
        torch.manual_seed(args.seed)
        self.manifold = make_manifold('Lorentz', c=args.c)
        self.input_dim = args.hidden_dim
        self.use_bias = args.bias
        self.cls = geoopt.ManifoldParameter(self.manifold.random_normal((args.n_classes, args.hidden_dim), std=1./math.sqrt(args.hidden_dim)), manifold=self.manifold, requires_grad=True)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False

    def decode(self, x, edge_index=None, edge_attr=None, idx=None):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias



class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, args):
        super(LinearDecoder, self).__init__()
        self.input_dim = args.hidden_dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls =  nn.Linear(self.input_dim, args.n_classes, bias=True)
        self.dropout = 0.0
        self.act = None

    def decode(self, x, edge_index=None, edge_attr=None, idx=None):
        out = self.cls(x)
        return out

model2decoder = {
    'GCN': GCNDecoder,
    'HyboNet': LorentzDecoder,
    'GAT': GATDecoder,
    'MLP': LinearDecoder,
    'HGCN': LinearDecoder
}