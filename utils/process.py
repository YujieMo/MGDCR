import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
import torch as th
import numpy as np
import dgl
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data, ClusterData,NeighborSampler
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.preprocessing import OneHotEncoder
import torch as th
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset
import pickle
from scipy.sparse import coo_matrix

def load_acm_mat(sc=3):
    data = sio.loadmat('data/acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp(sc=3):
    data = pkl.load(open("data/dblp.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["PAP"]
    adj_edge2 = data["PPrefP"]
    adj_edge3 = data["PATAP"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3



    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb(sc=3):
    data = pkl.load(open("data/imdb.pkl", "rb"))
    label = data['label']
###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    # for i in range(0,3550):
    #     for j in range(0, 3550):
    #         if adj_fusion[i][j]!=2:
    #             adj_fusion[i][j]=0
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc
    adj_fusion = adj_fusion  + np.eye(adj_fusion.shape[0])*3
    # torch.where(torch.eq(torch.Tensor(adj1), 1) == True)
    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    # adj1_dense = torch.dense(adj1)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot



def load_freebase(sc=3):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_m = sp.eye(type_num)
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    adj_list = [mam, mdm, mwm]
    adj_fusion = mam
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return  adj_list, feat_m, label, train[0], val[0], test[0], adj_fusion



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


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


