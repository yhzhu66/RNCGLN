import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon
import dgl
import os
import json

def load_single_graph(args=None, train_ratio=0.1, val_ratio=0.1):
    if args.dataset in ['CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if args.dataset in ['CiteSeer', 'PubMed']:
            # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            path = osp.join('./', 'dataset/')
            dataset = Planetoid(path, args.dataset)
        elif args.dataset in ['Photo', 'Computers']:
            # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            path = osp.join('./', 'dataset/')
            dataset = Amazon(path, args.dataset, pre_transform=None)  # transform=T.ToSparseTensor(),


        data = dataset[0]
        if 'train_mask' not in data:
            idx_train = range(int(0.05*data.x.size(0)))
            idx_val = range(int(0.05*data.x.size(0)), int(0.05*data.x.size(0))+int(0.1*data.x.size(0)))
            idx_test = range( int(0.65*data.x.size(0)), data.x.size(0))
            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
        else:
            idx_train = data.train_mask
            idx_val = data.val_mask
            idx_test = data.test_mask

        i = torch.Tensor.long(data.edge_index)
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        A = A_sp.to_dense()

        if args.gra_noi_rat !=0:
            A = sp.csr_matrix(A)
            import utils.random as UR
            attacker = UR.Random()
            n_perturbations = int(args.gra_noi_rat * (A.sum() // 2))
            perturbed_adj = attacker.attack(A, n_perturbations, type='flip')
            adj = sp.coo_matrix(perturbed_adj)
            values = adj.data
            indices = np.vstack((adj.row, adj.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adj.shape
            adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
            A =adj

        A_nomal = row_normalize(A)
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        A_I_nomal = row_normalize(A_I)
        label = data.y

        return [A_I_nomal, A_nomal, A], data.x, label, idx_train, idx_val, idx_test

    elif args.dataset in ['Cora']:
        idx_features_labels = np.genfromtxt("{}{}.content".format("./dataset/Cora/", "cora"), dtype=np.dtype(str))

        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        def normalize_features(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format("./dataset/Cora/", "cora"), dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        if args.gra_noi_rat !=0:
            import utils.random as UR
            attacker = UR.Random()
            A = adj
            n_perturbations = int(args.gra_noi_rat * (A.sum() // 2))
            perturbed_adj = attacker.attack(A, n_perturbations, type='flip')
            adj = sp.coo_matrix(perturbed_adj)


        features = normalize_features(features)
        A_I_nomal = normalize_adj(adj + sp.eye(adj.shape[0]))
        A_nomal = normalize_adj(adj)
        A =  adj.todense()

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        A_I_nomal = torch.FloatTensor(np.array(A_I_nomal.todense()))
        A_nomal = torch.FloatTensor(np.array(A_nomal.todense()))
        A = torch.FloatTensor(np.array(A))

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return [A_I_nomal, A_nomal, A], features, labels, idx_train, idx_val, idx_test

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features,eps =1e-6):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = rowsum + eps
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def row_normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch2dgl(graph):
    N = graph.shape[0]
    if graph.is_sparse:
        graph_sp = graph.coalesce()
    else:
        graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edges_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src, edges_dst), num_nodes=N)
    # graph_dgl.edate['w'] = edges_features
    return graph_dgl

