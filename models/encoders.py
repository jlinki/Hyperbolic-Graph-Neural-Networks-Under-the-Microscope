"""Graph encoders."""

import math

import torch
import torch.nn as nn
import manifolds

from layers.layers import GraphConvolution, MLP_layer, GraphAttention, HyboEuclideanConvolution
from layers.hyp_layers import LorentzGraphConvolution, HyperbolicGraphConvolution
from models.utils_models import make_manifold


class Encoder(nn.Module):
    """
    Abstract Encoder class.
    Compatible with nn.ModuleList, nn.Sequential, and nn.Module.
    Supports graph-based (x, edge_index) and non-graph-based (x) inputs.
    Automatically unpacks multiple outputs if present.
    """

    def __init__(self):
        super().__init__()
        self.layers = None  # to be set in subclass; None is a placeholder

    def encode(self, x, edge_index=None, edge_attr=None, return_all_layers=False):
        if self.layers is None:
            raise NotImplementedError("Encoder subclass must define self.layers.")
    
        # Helper function to call layer and unpack if tuple
        def call_layer(layer, *inputs):
            try:
                output = layer(*inputs) # PyG-style or standard nn.Module
            except TypeError:
                output = layer(inputs) # Tuple-style (e.g. (x,adj))

            if isinstance(output, tuple):
                return output[0]
            return output

        if isinstance(self.layers, nn.ModuleList):
            embedding_list = []
            for layer in self.layers:
                # Try calling with edge_index, else without
                try:
                    x = call_layer(layer, x, edge_index, edge_attr)
                except TypeError:
                    x = call_layer(layer, x)
                if return_all_layers:
                    embedding_list.append(x)
            if return_all_layers:
                return x, embedding_list
            else:
                return x, None
        # Single nn.Module or nn.Sequential
        try:
            return call_layer(self.layers, x, edge_index, edge_attr), None
        except TypeError:
            return call_layer(self.layers, x), None

    def forward(self, x, edge_index=None, edge_attr=None, return_all_layers=False):
        return self.encode(x, edge_index, edge_attr, return_all_layers)


class GCN(Encoder):

    def __init__(self, args):
        super(GCN, self).__init__()
        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(args.in_dim, args.hidden_dim, args.dropout, self.act))
        for _ in range(args.num_layers - 2):
            self.layers.append(GraphConvolution(args.hidden_dim, args.hidden_dim, args.dropout, self.act))
        if args.num_layers-1 > 0:
            if args.task == 'lp':
                self.layers.append(GraphConvolution(args.hidden_dim, args.hidden_dim, 0.0, self.act))
            else:
                self.layers.append(GraphConvolution(args.hidden_dim, args.hidden_dim, args.dropout, self.act))


class MLP(Encoder):

    def __init__(self, args):
        super(MLP, self).__init__()
        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None
        self.layers = nn.ModuleList()
        self.layers.append(MLP_layer(args.in_dim, args.hidden_dim, args.dropout, self.act))
        for _ in range(args.num_layers - 2):
            self.layers.append(MLP_layer(args.hidden_dim, args.hidden_dim, args.dropout, self.act))
        if args.num_layers-1 > 0:
            if args.task == 'lp':
                self.layers.append(MLP_layer(args.hidden_dim, args.hidden_dim, 0.0, self.act))
            else:
                self.layers.append(MLP_layer(args.hidden_dim, args.hidden_dim, args.dropout, self.act))


class GAT(Encoder):
    def __init__(self, args):
        super(GAT, self).__init__()
        # Choose activation
        if args.act is not None and args.act.lower() != "none":
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None

        self.layers = nn.ModuleList()

        # First layer
        first_layer = GraphAttention(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            act=self.act,
            num_heads=getattr(args, "num_heads", 1),
            concat=getattr(args, "concat", True),
        )
        self.layers.append(first_layer)

        # Middle layers
        in_dim = first_layer.out_dim()
        for _ in range(args.num_layers - 2):
            layer = GraphAttention(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                act=self.act,
                num_heads=getattr(args, "num_heads", 1),
                concat=getattr(args, "concat", True),
            )
            self.layers.append(layer)
            in_dim = layer.out_dim()

        # Last layer
        if args.num_layers - 1 > 0:
            last_dropout = 0.0 if args.task == "lp" else args.dropout
            last_layer = GraphAttention(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                dropout=last_dropout,
                act=self.act,
                num_heads=getattr(args, "num_heads", 1),
                concat=getattr(args, "concat", True)
            )
            self.layers.append(last_layer)


class HyboNet(Encoder):

    def __init__(self, args):
        super(HyboNet, self).__init__()
        self.manifold = make_manifold(args.manifold, args.c)
        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None
        self.layers = nn.ModuleList()
        self.layers.append(LorentzGraphConvolution(self.manifold, args.in_dim +1 , args.hidden_dim, True, args.dropout,  args.use_att, None))
        for _ in range(args.num_layers - 2):
            self.layers.append(LorentzGraphConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, args.dropout,  args.use_att, self.act))
        if args.num_layers-1 > 0:
            if args.task == 'lp':
                self.layers.append(LorentzGraphConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, 0.0,  args.use_att, self.act))
            else:
                self.layers.append(LorentzGraphConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, args.dropout,  args.use_att, self.act))


class HyboNetEuclidean(Encoder):

    def __init__(self, args):
        super(HyboNetEuclidean, self).__init__()
        self.manifold = make_manifold(args.manifold, args.c)

        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None
        self.layers = nn.ModuleList()
        self.layers.append(HyboEuclideanConvolution(self.manifold, args.in_dim, args.hidden_dim, True, args.dropout,  args.use_att, None))
        for _ in range(args.num_layers - 2):
            self.layers.append(HyboEuclideanConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, args.dropout,  args.use_att, self.act))
        if args.num_layers-1 > 0:
            if args.task == 'lp':
                self.layers.append(HyboEuclideanConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, 0.0,  args.use_att, self.act))
            else:
                self.layers.append(HyboEuclideanConvolution(self.manifold, args.hidden_dim , args.hidden_dim, True, args.dropout,  args.use_att, self.act))


class HGCN(Encoder):

    def __init__(self, args):
        super(HGCN, self).__init__()

        # self.manifolds = nn.ModuleList([
        #     make_manifold(args.manifold, args.c, learnable=True) for _ in range(args.num_layers + 1)
        # ])
        if args.c_learnable:
            self.manifolds = nn.ModuleList([
                make_manifold(args.manifold, args.c, learnable=True) for _ in range(args.num_layers + 1)
            ])
        else:
            self.manifolds = nn.ModuleList([
                make_manifold(args.manifold, args.c, learnable=False) for _ in range(args.num_layers + 1)
            ])


        if args.act is not None and args.act.lower() != 'none':
            self.act = getattr(torch.nn.functional, args.act)
        else:
            self.act = None

        self.layers = nn.ModuleList()


        if args.num_layers > 1 :        
            self.layers.append(HyperbolicGraphConvolution(self.manifolds[0], self.manifolds[1], args.in_dim, args.hidden_dim, args.dropout, self.act, args.bias, args.use_att, args.local_agg))
        else:
            self.layers.append(HyperbolicGraphConvolution(self.manifolds[0], self.manifolds[0], args.in_dim, args.hidden_dim, args.dropout, self.act, args.bias, args.use_att, args.local_agg))
        
        for i in range(args.num_layers - 2):
            self.layers.append(HyperbolicGraphConvolution(self.manifolds[i+1], self.manifolds[i+2], args.hidden_dim, args.hidden_dim, args.dropout, self.act, args.bias, args.use_att, args.local_agg))
        if args.num_layers-1 > 0:
            if args.task == 'lp':
                self.layers.append(HyperbolicGraphConvolution(self.manifolds[-2], self.manifolds[-1], args.hidden_dim, args.hidden_dim, 0.0, self.act, args.bias, args.use_att, args.local_agg))
            else:
                self.layers.append(HyperbolicGraphConvolution(self.manifolds[-2], self.manifolds[-1], args.hidden_dim, args.hidden_dim, args.dropout, self.act, args.bias, args.use_att, args.local_agg))







