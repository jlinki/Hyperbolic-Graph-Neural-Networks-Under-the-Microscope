import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAtt(nn.Module):
    """Dense attention layer. Complexity is O(n^2*d)."""
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        x_left = torch.unsqueeze(x, 1) # shape [n, 1, d]
        x_left = x_left.expand(-1, n, -1) # shape [n, n, d]
        x_right = torch.unsqueeze(x, 0) # shape [1, n, d]
        x_right = x_right.expand(n, -1, -1) # shape [n, n, d]

        x_cat = torch.cat((x_left, x_right), dim=2) # shape [n, n, 2d] 
        att_adj = self.linear(x_cat).squeeze() # shape [n, n]
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj) # shape [n, n]

        return att_adj
    


class SparseAtt(nn.Module):
    """Sparse attention layer. Complexity is O(E*d)."""
    def __init__(self, in_features, dropout):
        super(SparseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        if adj.is_sparse:
            adj = adj.coalesce() #sparse coalesced form 
            edge_index = adj._indices() # shape [2, E], e.g., [[0,1,1],[1,0,2]] means edges (0,1),(1,0),(1,2)
        else:
            edge_index = adj.nonzero().t()

        row, col = edge_index

        x_i, x_j = x[row], x[col]  #x_i features of source nodes, x_j features of target nodes

        w_ij = torch.cat([x_i, x_j], dim=1)  # shape [E, 2d]
        m_ij = self.linear(w_ij).squeeze()  # shape [E]
        att_ij = torch.sigmoid(m_ij)  # shape [E]

        #softmax normalization 
        # exp_m_ij = torch.exp(m_ij)
        # sum_per_row = scatter(exp_m_ij, row, dim=0, reduce='sum')  # [E]
        # att_ij = exp_m_ij / (sum_per_row[row] + 1e-16)  # [E]

        return att_ij, x_i, x_j, row, col
    