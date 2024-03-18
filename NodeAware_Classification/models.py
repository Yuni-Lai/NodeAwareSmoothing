import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops,to_dense_adj
from torch_sparse import spmm
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from utils import *

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout=dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_remaining_self_loops(edge_index)
        x = self.conv1(features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout,training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SmoothGCN(GCN):
    '''Multi-smoothing Smoothing model for GCN. Also suitable for other GNNs.'''
    def __init__(self,in_channels, out_channels, hidden_channels, dropout,config,device):
        super(SmoothGCN, self).__init__(in_channels, out_channels, hidden_channels, dropout)
        self.config = config
        self.device=device
        self.nclass=out_channels
        self.p_e, self.p_n = torch.tensor(config['p_e']), torch.tensor(config['p_n'])

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense)!=torch.tril(adj_dense).t()).sum()==0
        adj_triu = torch.triu(adj_dense,diagonal=1)
        adj_triu = (adj_triu==1) * torch.bernoulli(torch.ones(size).to(self.device)*(1 - self.p_e))
        # deleted edges: (torch.triu(adj_dense)>0).sum()-(adj_triu > 0).sum()
        adj_triu = adj_triu.mul(torch.bernoulli(torch.ones(size[0]).to(self.device)*(1 - self.p_n)))
        # deleted nodes: (torch.bernoulli(torch.ones(size[0]).to(self.device)*(1 - self.p_n))==0).sum()
        adj_perted = adj_triu + adj_triu.t()
        # total deleted edges: (torch.triu(adj_dense)>0).sum()-(torch.triu(adj_perted) > 0).sum()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            adj_dense = torch.squeeze(to_dense_adj(edge_index))
            adj_dense = self.perturbation(adj_dense)
            edge_index = torch.nonzero(adj_dense).t()
            # deleted_nodes=[]
            # for i in range(adj_dense.shape[0]):
            #     if (adj_dense[i,:]==0).all():
            #         deleted_nodes.append(i)
            # len(deleted_nodes)
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2]
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2




class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        edge_index, _ = add_remaining_self_loops(edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SmoothGAT(GAT):
    def __init__(self, in_channels, out_channels, hidden_channels,heads, dropout, config, device):
        super(SmoothGAT, self).__init__(in_channels, out_channels, hidden_channels, heads, dropout)
        self.config = config
        self.device = device
        self.nclass = out_channels
        self.p_e, self.p_n = torch.tensor(config['p_e']), torch.tensor(config['p_n'])

    def perturbation(self, adj_dense):
        '''Using upper triangle adjacency matrix to Perturb the edge first, and then the nodes.'''
        size = adj_dense.shape
        assert (torch.triu(adj_dense) != torch.tril(adj_dense).t()).sum() == 0
        adj_triu = torch.triu(adj_dense, diagonal=1)
        adj_triu = (adj_triu == 1) * torch.bernoulli(torch.ones(size).to(self.device) * (1 - self.p_e))
        # deleted edges: (torch.triu(adj_dense)>0).sum()-(adj_triu > 0).sum()
        adj_triu = adj_triu.mul(torch.bernoulli(torch.ones(size[0]).to(self.device) * (1 - self.p_n)))
        # deleted nodes: (torch.bernoulli(torch.ones(size[0]).to(self.device)*(1 - self.p_n))==0).sum()
        adj_perted = adj_triu + adj_triu.t()
        # total deleted edges: (torch.triu(adj_dense)>0).sum()-(torch.triu(adj_perted) > 0).sum()
        return adj_perted

    def forward_perturb(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """ Estimate the model with smoothing perturbed samples """
        # features: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph adjacency matrix of shape [2, num_edges]
        with torch.no_grad():
            adj_dense = torch.squeeze(to_dense_adj(edge_index))
            adj_dense = self.perturbation(adj_dense)
            edge_index = torch.nonzero(adj_dense).t()
            # deleted_nodes=[]
            # for i in range(adj_dense.shape[0]):
            #     if (adj_dense[i,:]==0).all():
            #         deleted_nodes.append(i)
            # len(deleted_nodes)
        return self.forward(features, edge_index)

    def smoothed_precit(self, features, edge_index, num):
        """ Sample the base classifier's prediction under smoothing perturbation of the input x.
        num: number of samples to collect (N)
        return: top2: the top 2 classes, and the per-class counts
        """
        counts = np.zeros((features.shape[0], self.nclass), dtype=int)
        for i in tqdm(range(num), desc='Processing MonteCarlo'):
            predictions = self.forward_perturb(features, edge_index).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), self.nclass)
        top2 = counts.argsort()[:, ::-1][:, :2]
        count1 = [counts[n, idx] for n, idx in enumerate(top2[:, 0])]
        count2 = [counts[n, idx] for n, idx in enumerate(top2[:, 1])]
        return top2, count1, count2
