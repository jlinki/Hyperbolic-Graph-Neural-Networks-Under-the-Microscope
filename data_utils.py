import os
import torch
import random
import inspect
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, to_scipy_sparse_matrix, add_self_loops, remove_self_loops
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

### Utility functions for loading and processing graph datasets

def create_random_binary_balanced_splits(data, val_prop=0.1, test_prop=0.1, seed=42):
    """
    Given a PyG Data object with a 'y' attribute (labels),
    generate 10 random train/val/test splits and attach them
    as concatenated boolean masks to the Data object.
    The test and val splits are 50-50 balanced.

    Each mask will have shape [num_nodes, 10].

    Args:
        data (torch_geometric.data.Data): PyG Data object containing `y`.
        val_prop (float): Proportion of data for validation.
        test_prop (float): Proportion of data for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        torch_geometric.data.Data: The same Data object with
        `train_mask`, `val_mask`, and `test_mask` attributes of shape [num_nodes, 10].
    """
    np.random.seed(seed)
    labels = data.y.cpu().numpy()
    n_nodes = labels.shape[0]

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    train_masks, val_masks, test_masks = [], [], []

    for split_id in range(10):
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        # Determine counts per class
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        n_val_pos = round(val_prop * 0.5 * n_nodes)
        n_test_pos = round(test_prop * 0.5 * n_nodes)
        n_val_neg = round(val_prop * 0.5 * n_nodes)
        n_test_neg = round(test_prop * 0.5 * n_nodes)

        assert n_val_pos + n_test_pos <= min(n_pos, n_neg), "Not enough samples to create the requested splits."

        # Split indices
        idx_val = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
        idx_test = np.concatenate([
            pos_idx[n_val_pos:n_val_pos + n_test_pos],
            neg_idx[n_val_neg:n_val_neg + n_test_neg]
        ])
        idx_train = np.concatenate([
            pos_idx[n_val_pos + n_test_pos:],
            neg_idx[n_val_neg + n_test_neg:]
        ])

        # Boolean masks
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        train_masks.append(train_mask.unsqueeze(1))
        val_masks.append(val_mask.unsqueeze(1))
        test_masks.append(test_mask.unsqueeze(1))

    # Concatenate all 10 splits along dim=1 → [num_nodes, 10]
    data.train_mask = torch.cat(train_masks, dim=1)
    data.val_mask = torch.cat(val_masks, dim=1)
    data.test_mask = torch.cat(test_masks, dim=1)

    return data


def create_random_binary_splits(data, val_prop=0.1, test_prop=0.1, seed=42):
    """
    Given a PyG Data object with a 'y' attribute (labels),
    generate 10 random train/val/test splits and attach them
    as concatenated boolean masks to the Data object.
    All splits keep the ratio of class 1/0's of the original data.

    Each mask will have shape [num_nodes, 10].

    Args:
        data (torch_geometric.data.Data): PyG Data object containing `y`.
        val_prop (float): Proportion of data for validation.
        test_prop (float): Proportion of data for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        torch_geometric.data.Data: The same Data object with
        `train_mask`, `val_mask`, and `test_mask` attributes of shape [num_nodes, 10].
    """
    np.random.seed(seed)
    labels = data.y.cpu().numpy()
    n_nodes = labels.shape[0]

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    train_masks, val_masks, test_masks = [], [], []

    for split_id in range(10):
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        # Determine counts per class
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        n_val_pos = round(val_prop * n_pos)
        n_test_pos = round(test_prop * n_pos)
        n_val_neg = round(val_prop * n_neg)
        n_test_neg = round(test_prop * n_neg)

        # Split indices
        idx_val = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
        idx_test = np.concatenate([
            pos_idx[n_val_pos:n_val_pos + n_test_pos],
            neg_idx[n_val_neg:n_val_neg + n_test_neg]
        ])
        idx_train = np.concatenate([
            pos_idx[n_val_pos + n_test_pos:],
            neg_idx[n_val_neg + n_test_neg:]
        ])

        # Boolean masks
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        train_masks.append(train_mask.unsqueeze(1))
        val_masks.append(val_mask.unsqueeze(1))
        test_masks.append(test_mask.unsqueeze(1))

    # Concatenate all 10 splits along dim=1 → [num_nodes, 10]
    data.train_mask = torch.cat(train_masks, dim=1)
    data.val_mask = torch.cat(val_masks, dim=1)
    data.test_mask = torch.cat(test_masks, dim=1)

    return data


def ensure_bool_masks(data):
    for key in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, key):
            mask = getattr(data, key)
            if mask.dtype != torch.bool:
                setattr(data, key, mask.bool())
    return data


def filter_kwargs(func, kwargs):
    """Return only the kwargs that match the parameters of the given function."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def load_data_airport(data_path, task, augment_features=True):
    path = os.path.join(data_path, 'airport/airport.p')
    with open(path, 'rb') as f:
        graph = pkl.load(f)

    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    row, col = [], []
    for u, v in graph.edges():
        if u == v:  # skip self-loops
            continue
        else:
            row.append(node_to_idx[u])
            col.append(node_to_idx[v])
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)).tocsr()

    features = np.array([graph.nodes[u]['feat'] for u in nodes])

    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    if augment_features is True:
        if task == 'lp':
            adj_sym = adj # keep original adjacency for link prediction with undirected edges
        else:
            adj_sym = adj + adj.T # we make adjacency symmetric for node classification features
            adj_sym[adj_sym > 1] = 1  # in case of multiple
        deg = np.squeeze(np.sum(adj_sym, axis=0).astype(int))
        deg = np.clip(deg, 0, 5)  # Clamp values >5 to 5
        deg_onehot = np.eye(6)[deg].squeeze()  # One-hot encode
        const_f = np.ones((features.shape[0], 1))
        features = np.concatenate((features, deg_onehot, const_f), axis=1)
    labels = np.digitize(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    return adj, features, labels


def load_synthetic_data(dataset_str, data_dir, use_feats=True):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    data_path = os.path.join(data_dir, dataset_str)
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
        if dataset_str == 'disease_lp_new':
            features = features[:, :-1]  # remove last branch feature
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str))) # labels = np.load(os.path.join(data_path, "{}.labels_norm.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


#### FEATURE PROCESSING ####

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


### Main class for loading graph datasets

class GraphDatasetLoader:
    def __init__(self, dataset_name, task='nc', data_dir='./data',
                 split_seed=42, *args, **kwargs):   
        assert task in ['nc', 'nr', 'lp', 'pairs']
        self.dataset_name = dataset_name.lower()
        self.task = task
        self.data_dir = data_dir
        self.split_seed = split_seed
        self.args = args
        self.all_kwargs = kwargs
        self.permuted = False
        self.synthetic_datasets = ['tree1111_g00_lp', 'tree1111_g005_lp', 'tree1111_g010_lp','tree1111_g015_lp', 'tree1111_g02_lp',
                                'tree1111_g04_lp', 'tree1111_g06_lp', 'tree1111_g08_lp', 'tree1111_g10_lp',
                                'tree100_2_2_2_nc', 'disease_lp', 'disease_nc',
                                   'disease_m_lp', 'pdp_tree','pdp_grid', 'grid_28x28_nr', 'tree100_2_2_2_null_nc']


    def load(self):
        if self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            return self._load_builtin_dataset()
        elif self.dataset_name == 'airport':
            return self._load_airport_dataset()
        elif self.dataset_name in self.synthetic_datasets:
            return self._load_custom_dataset()
        elif self.dataset_name in ["roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"]:
            return self._load_heterophilious_dataset()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported or not recognized.")


    def _load_builtin_dataset(self):
        planetoid_kwargs = filter_kwargs(Planetoid, self.all_kwargs)
        if self.task in ['nc', 'nr']:
            planetoid_kwargs['split'] = 'public'
        dataset = Planetoid(root=os.path.join(self.data_dir, self.dataset_name), name=self.dataset_name.capitalize(), **planetoid_kwargs)
        data = dataset[0]
        data = ensure_bool_masks(data)

        # Process features and adjacency matrix
        if self.all_kwargs.get('normalize_feats', True):
            data.x = torch.tensor(normalize(data.x.numpy()), dtype=torch.float)
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        if self.all_kwargs.get('normalize_adj', True):
            adj = adj + sp.eye(adj.shape[0])
            adj = normalize(adj)  # row-normalize
        else:
            if self.task == 'lp':
                adj = adj + sp.eye(adj.shape[0])  # we always add self-loops for link prediction and remove them later if necessary
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        data.edge_index, data.edge_attr = edge_index, edge_weight.float()

        if self.task == 'lp':
            data = self._apply_link_split(data)
        elif self.task in ['nc', 'nr']:
            data = self._apply_node_split(data)
        return data


    def _load_airport_dataset(self):
        adj, features, labels = load_data_airport(self.data_dir, task=self.task)
        # Process features and adjacency matrix
        if self.all_kwargs.get('normalize_feats', True):
            features = normalize(features)
        if self.task in ['nc', 'nr']:
            # make adjacency symmetric for node classification
            adj = adj + adj.T
            adj[adj > 1] = 1  # in case of multiple edges
            if self.all_kwargs.get('normalize_adj', True):
                adj = adj + sp.eye(adj.shape[0])
                adj = normalize(adj)
        elif self.task == 'lp':
            adj = adj + adj.T
            adj[adj > 1] = 1  # in case of multiple edges
            if self.all_kwargs.get('normalize_adj', True):
                adj = adj + sp.eye(adj.shape[0])
                adj = normalize(adj)
            else:
                adj = adj + sp.eye(adj.shape[0]) # we always add self-loops for link prediction and remove them later if necessary

        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        #edge_index = to_undirected(edge_index)

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float(), y=y)

        if self.task in ['nc', 'nr']:
            return self._apply_node_split(data)
        else:
            return self._apply_link_split(data)


    def _load_custom_dataset(self):
        if self.dataset_name in ['tree100_2_2_2_null_nc']:
            self.dataset_name = 'tree100_2_2_2_nc'
            adj, features, labels = load_synthetic_data(self.dataset_name, self.data_dir)
            np.random.seed(42)
            permutation = np.random.permutation(labels.shape[0])
            labels = labels[permutation]
        else:
            adj, features, labels = load_synthetic_data(self.dataset_name, self.data_dir)

        # Process features and adjacency matrix
        if self.all_kwargs.get('normalize_feats', True):
            features = normalize(features)
        if self.all_kwargs.get('normalize_adj', True):
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            if self.task == 'lp':
                adj = adj + sp.eye(adj.shape[0])

        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        x = torch.tensor(features.todense(), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float(), y=y)

        if self.task in ['nc', 'nr']:
            if self.dataset_name == 'disease_nc':
                return create_random_binary_splits(data, val_prop=self.all_kwargs.get('num_val', 0.15), test_prop=self.all_kwargs.get('num_test', 0.15), seed=self.split_seed)
            else:
                return self._apply_node_split(data)
        elif self.task == 'lp':
            return self._apply_link_split(data)
        else: 
            return self._apply_pairs_split(data)
        

    def _load_heterophilious_dataset(self):
        het_kwargs = filter_kwargs(HeterophilousGraphDataset, self.all_kwargs)
        dataset = HeterophilousGraphDataset(root=os.path.join(self.data_dir, self.dataset_name), name=self.dataset_name, **het_kwargs)
        data = dataset[0]
        if self.task == 'lp':
            data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr)
            data = self._apply_link_split(data)
        return data


    def _apply_link_split(self, data):
        link_kwargs = filter_kwargs(RandomLinkSplit, self.all_kwargs)

        # Set random seeds for reproducibility
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        random.seed(self.split_seed)

        # remove masks and labels
        data.train_mask = data.val_mask = data.test_mask = None
        data.y = None

        # possible parameters for RandomLinkSplit
        # is_undirected = False,
        # split_labels = False,
        # num_val = 0.1,
        # num_test = 0.2,
        # add_negative_train_samples = True,
        # neg_sampling_ratio = 1.0,
        # disjoint_train_ratio = 0.0
        # edge_types = None,
        # rev_edge_types = None,
        num_splits = 10
        splits = []

        # isolate self-loops
        edges_with_self_loops, edge_attr_with_self_loops = data.edge_index, data.edge_attr
        edges_self_loops = edges_with_self_loops[:, edges_with_self_loops[0, :] == edges_with_self_loops[1, :]]
        edge_attr_self_loops = edge_attr_with_self_loops[edges_with_self_loops[0, :] == edges_with_self_loops[1, :]]
        edges_without_self_loops, edge_attr_without_self_loops = remove_self_loops(edges_with_self_loops, edge_attr_with_self_loops)
        edge_index, edge_attr = edges_without_self_loops, edge_attr_without_self_loops

        # sort edges (required by RandomLinkSplit)
        edges = edge_index.t()
        sorted_idx = torch.argsort(edges[:, 0] * edges[:, 0].max() + edges[:, 1])
        edges = edges[sorted_idx]
        edge_index_sorted = edges.t()
        edge_attr_sorted = edge_attr[sorted_idx]
        data.edge_index, data.edge_attr = edge_index_sorted, edge_attr_sorted
        for i in range(num_splits):
            # Set random seeds for reproducibility
            transform = RandomLinkSplit(is_undirected=True, split_labels=True, **link_kwargs)
            train_data, val_data, test_data = transform(data)
            curr_data = {"train": train_data, "val": val_data, "test": test_data}
            for key in curr_data.keys():
                if self.all_kwargs.get('add_self_loops', False):
                    curr_data[key].edge_index = torch.cat([curr_data[key].edge_index, edges_self_loops], dim=1)
                    if self.all_kwargs.get('normalize_adj', True):
                        adj = to_scipy_sparse_matrix(curr_data[key].edge_index, num_nodes=data.num_nodes)
                        adj = normalize(adj)
                        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
                        curr_data[key].edge_attr = edge_attr
                    else:
                        curr_data[key].edge_attr = torch.cat([curr_data[key].edge_attr, torch.ones_like(edge_attr_self_loops)], dim=0)
                pos_edge_label_index = to_undirected(curr_data[key].pos_edge_label_index)
                neg_edge_label_index = to_undirected(curr_data[key].neg_edge_label_index)
                curr_data[key].edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
                curr_data[key].edge_label = torch.cat([torch.ones(pos_edge_label_index.size(1), dtype=torch.long),
                                                      torch.zeros(neg_edge_label_index.size(1), dtype=torch.long)], dim=0)

            splits.append(curr_data)
        return splits


    def _apply_node_split(self, data):

        # Set random seeds for reproducibility
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        random.seed(self.split_seed)

        node_kwargs = filter_kwargs(RandomNodeSplit, self.all_kwargs)
        # Set random seeds for reproducibility
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        random.seed(self.split_seed)

        transform = RandomNodeSplit(num_splits= 10, **node_kwargs)

        return transform(data)




    def _apply_pairs_split(self, data, num_splits=10):
        """
        Split an n x n label matrix into train/val/test pairs.
        Returns a list of dictionaries, each dict contains PyG Data objects
        for 'train', 'val', 'test'.
        """
        n = data.y.shape[0]  # assume data.y is n x n
        # retain only the upper triangular part because the matrix is symmetric
        all_pairs = np.array([[i, j] for i in range(n) for j in range(i+1, n)])

        splits = []

        y = data.y.clone()

        # build labels ALIGNED with shuffled pairs
        all_labels = torch.tensor(
            [y[i, j] for i, j in all_pairs],
            dtype=torch.float
        )

        # NULL: permute labels only
        if self.permuted:
            perm_gen = torch.Generator()
            perm_gen.manual_seed(42)
            perm = torch.randperm(all_labels.numel(), generator=perm_gen, device=all_labels.device)
            all_labels_perm = all_labels[perm].clone()

            corr = torch.corrcoef(
                torch.stack([all_labels.float(), all_labels_perm.float()])
            )[0, 1].item()
            print(f"corr(labels_true, labels_perm) = {corr:.3f}")

            all_labels = all_labels_perm

            # overwrite upper triangle
            for (i, j), val in zip(all_pairs, all_labels):
                y[i, j] = val
                y[j, i] = val


        for split_idx in range(num_splits):
            seed = self.split_seed + split_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            np.random.shuffle(all_pairs)

            n_train = int(0.7 * len(all_pairs))
            n_val   = int(0.15 * len(all_pairs))

            train_pairs = all_pairs[:n_train]
            val_pairs   = all_pairs[n_train:n_train+n_val]
            test_pairs  = all_pairs[n_train+n_val:]


            def pairs_to_tensor(pairs):
                return torch.tensor(pairs, dtype=torch.long).t()  # [2, num_pairs]
    

            def labels_tensor(pairs, y):
                return torch.tensor([y[i, j] for i, j in pairs], dtype=torch.float)



            split_data = {
                "train": Data(
                    x=data.x.clone(),
                    edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone(),
                    pos_edge_label_index=pairs_to_tensor(train_pairs),
                    pos_edge_label=labels_tensor(train_pairs, y)
                ),
                "val": Data(
                    x=data.x.clone(),
                    edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone(),
                    pos_edge_label_index=pairs_to_tensor(val_pairs),
                    pos_edge_label=labels_tensor(val_pairs, y)
                ),
                "test": Data(
                    x=data.x.clone(),
                    edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone(),
                    pos_edge_label_index=pairs_to_tensor(test_pairs),
                    pos_edge_label=labels_tensor(test_pairs, y)
                ),
            }

            splits.append(split_data)

        return splits

