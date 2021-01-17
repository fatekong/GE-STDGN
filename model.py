#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter

cuda_gpu = torch.cuda.is_available()

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)

class DenseGNN(nn.Module):
    def __init__(self, in_features, out_feature, gdep, dropout, adj, device):
        super(DenseGNN, self).__init__()
        self.nconv = nconv()
        self.mlp = []
        for i in range(gdep):
            if i != gdep-1:
                self.mlp.append(linear((i+2) * in_features, in_features))
            else:
                self.mlp.append(linear((i + 2) * in_features, out_feature))
        self.gdep = gdep
        self.dropout = dropout
        self.device = device
        self.adj = adj

    def forward(self, x, h):
        x = torch.cat((x, h), dim=2)
        x = x.transpose(1, 2).unsqueeze(3)
        x = x.contiguous()
        d = self.adj.sum(1)
        h = x.clone()
        # out = [h]
        # a = self.adj / d.view(-1, 1)
        # for i in range(self.gdep):
        #     h = self.nconv(h, a)
        #     out.append(h)
        #     ho = torch.cat(out, dim=1)
        # # print('ho', ho.shape)
        #     ho = self.mlp[i](ho).squeeze(3).transpose(2, 1)
        a = self.adj / d.view(-1, 1)
        dense = []
        for i in range(self.gdep):
            h_a = self.nconv(h, a)
            dense.append(h_a)
            out = torch.cat((h, torch.cat(dense, dim=1)), dim=1)
            h = self.mlp[i](out)
        h = h.squeeze(3).transpose(2, 1)
        return h

# DenseGNN Cell is used in processing spatial relations in STDGN
class DGNCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, device, deep):
        super(DGNCell, self).__init__()
        self.nodes = num_nodes
        self.units = num_units
        # 从TGC网络出来时的特征数目
        if cuda_gpu:
            _adj = adj.to(device)
        self.gcn_1 = DenseGNN(2 * num_units, 2 * self.units, gdep=deep, dropout=0.1, adj=_adj, device=device)
        self.gcn_2 = DenseGNN(2 * num_units, self.units, gdep=deep, dropout=0.1, adj=_adj, device=device)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.A = Parameter(_adj)
        # self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, X, state=None):
        if state is None:
            state = X
        graph_value = self.sigmoid(self.gcn_1(X, state))
        r, z = graph_value.chunk(2, dim=2)
        r_state = r * state
        h_t1 = self.tan(self.gcn_2(X, r_state))
        # 源代码中的GRU更新代码是错误的，如下
        new_h = z * state + (1 - z) * h_t1
        # print("new_h.shape:", new_h.shape)
        # 正确的应该是：
        # new_h = z * h_t1 + (1-z) * state
        return new_h, new_h

class AttEncoderLayer(nn.Module):
    def __init__(self, num_units, seq_len, device):
        super(AttEncoderLayer, self).__init__()
        self.device = device
        self.attn = nn.Linear(num_units, seq_len).to(device)
        self.hidden_size = num_units
        self.relu = nn.ReLU()

    def reset_parameters(self):
        attn_weight = self.attn.weight
        attn_bias = self.attn.bias
        torch.nn.init.xavier_uniform_(attn_weight, gain=1)
        torch.nn.init.constant_(attn_bias, 0)

    def forward(self, seq_list):
        seq = seq_list.shape[2]
        nodes = seq_list.shape[1]
        batch_size = seq_list.shape[0]
        x = seq_list[:, :, -1, :]

        attn_weights = F.softmax(self.relu(self.attn(x)), dim=2)
        attn_applied = torch.bmm(
            attn_weights.view(attn_weights.shape[0] * attn_weights.shape[1], 1, attn_weights.shape[2]),
            seq_list.view(seq_list.shape[0] * seq_list.shape[1], seq, self.hidden_size)).squeeze(1)
        attn_applied = attn_applied.view(batch_size, nodes, attn_applied.shape[1])
        return attn_applied

# This model is a STDGN model without multi-head attention mechanism
# The reason way attention dose not used is to save resources and reduce computing time

class STDGN_woa(nn.Module):
    name = 'STDGN_woa'
    #seq_len is input steps, pre_len is output steps
    def __init__(self, num_units, adj, num_nodes, num_feature,
                  seq_len, pre_len, device, deep=1):
        super(STDGN_woa, self).__init__()
        self.seq = seq_len
        self.nodes = num_nodes
        self.dgncell = DGNCell(num_units, adj, num_nodes, device, deep)
        self.seq_linear = nn.Linear(num_feature, num_units)
        self.drop = nn.Dropout(0.2)
        #self.sigmoid = nn.Sigmoid()
        self.pre_linear = nn.Linear(num_units, pre_len)

    def forward(self, x):
        #x = [batchsize, nodes, seq_len, feature]
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[3])
        input = self.drop(self.seq_linear(x))
        #x = [batchsize*nodes*seq_len, feature]
        input = input.view(batch_size, self.nodes, self.seq, input.shape[-1])
        seq_list = []
        for i in range(self.seq):
            if i == 0:
                x, h = self.dgncell(input[:, :, i, :])
            else:
                x, h = self.dgncell(input[:, :, i, :], h)
            seq_list.append(x)
        #last_list = [batch_size * nodes, hidden]
        last_list = seq_list[-1]
        output = self.pre_linear(last_list)
        output = output.view(batch_size, self.nodes, -1)
        return output

class STDGN(nn.Module):
    name = "STDGN"
    def __init__(self, num_units, adj, num_nodes, num_feature, num_head,
                  seq_len, pre_len, device, deep=1):
        super(STDGN, self).__init__()
        self.device = device
        self.seq = seq_len
        self.nodes = num_nodes
        self.head = num_head
        self.hidden_size = num_units


        # multi-head attention
        self.attn = [AttEncoderLayer(num_units, seq_len, device) for _ in range(num_head)]
        self.attn_combine = nn.Linear(2*num_units, pre_len)

        self.dgncell = DGNCell(num_units, adj, num_nodes, device, deep)
        self.seq_linear = nn.Linear(num_feature, num_units)

        self.drop = nn.Dropout(0.1)
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        #attn_weight = self.attn.weight
        #attn_bias = self.attn.bias
        attn_c_weight = self.attn_combine.weight
        attn_c_bias = self.attn_combine.bias
        #torch.nn.init.xavier_uniform_(attn_weight, gain=1)
        #torch.nn.init.constant_(attn_bias, 0)
        torch.nn.init.xavier_uniform_(attn_c_weight, gain=1)
        torch.nn.init.constant_(attn_c_bias, 0)

    def forward(self, x):
        # x = [batchsize, nodes, seq_len, feature]
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[3])
        input = self.drop(self.seq_linear(x))
        # x = [batchsize*nodes*seq_len, feature]
        input = input.view(batch_size, self.nodes, self.seq, input.shape[-1])
        for i in range(self.seq):
            if i == 0:
                x, h = self.dgncell(input[:, :, i, :])
                seq_list = x.unsqueeze(2)
            else:
                x, h = self.tgcncell(input[:, :, i, :], h)
                seq_list = torch.cat((seq_list, x.unsqueeze(2)), dim=2)
        for i in range(len(self.attn)):
            if i == 0:
                attn_applied = self.attn[i](seq_list)
            else:
                attn_applied += self.attn[i](seq_list)
        # average multi-head attention
        attn_applied = attn_applied/len(self.attn)

        # concat multi-head attention
        # attn_applied = torch.cat([att(seq_list) for att in self.attn], dim=2)
        output = torch.cat((seq_list[:, :, -1, :], attn_applied), dim=2).view(batch_size * self.nodes, -1)
        output = self.attn_combine(output).view(batch_size, self.nodes, -1)
        return output

# ------------------------------------------- other graph cell used in processing spatial relation _________________________________________

# Those models are not designed by us, but adopted in GNN cell to compare in the experiments.
# And the source of their article will be noted below.

# Spectral-based GCN Cell is used in processing spatial relations
# This model obtained from paper: https://arxiv.org/abs/2006.11583
class GCNforGRU(nn.Module):
    def __init__(self, in_features, out_features, adj, num_nodes, bias=False):
        super(GCNforGRU, self).__init__()
        self.nodes = num_nodes
        self.adj = adj
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        ## x ->(batchsize, node_num, feature)
        ## h ->(batchsize, node_num, feature)

        input = torch.cat((x, h), dim=2)

        feature_size = input.shape[2]

        input = input.permute(1, 2, 0)
        input = input.reshape(self.nodes, -1)

        output = torch.matmul(self.adj, input)
        output = output.view(self.nodes, feature_size, -1)

        output = output.permute(2, 0, 1)
        output = output.reshape(output.shape[0]*output.shape[1], -1)

        output = torch.matmul(output, self.weight)

        if self.bias is not None:
            output += self.bias

        output = output.view(-1, self.nodes, output.shape[1])
        return output

class GCNCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, device):
        super(GCNCell, self).__init__()
        self.nodes = num_nodes
        self.units = num_units

        # Normalization of graph volume product based on spectrum to obtain adjacency matrix
        _adj = self.gen_adj(adj)
        if cuda_gpu:
            _adj = _adj.to(device)

        self.gcn_1 = GCNforGRU(2*num_units, 2*self.units, _adj, num_nodes)
        self.gcn_2 = GCNforGRU(2*num_units, self.units, _adj, num_nodes)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.A = Parameter(_adj)

    def gen_adj(self, A):
        D = torch.pow(A.sum(1), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, X, state=None):
        if state is None:
            state = X
        graph_value = self.sigmoid(self.gcn_1(X, state))
        r, z = graph_value.chunk(2, dim=2)
        r_state = r * state
        h_t1 = self.tan(self.gcn_2(X, r_state))
        new_h = z * state + (1-z) * h_t1
        return new_h, new_h

# Spatial-based GAT Cell is used in processing spatial relations
class GATCell(nn.Module):
    def __init__(self, num_units, adj, num_nodes, dropout, alpha, device, sparse=False):
        super(GATCell, self).__init__()

        # 'sparse' is the decision applied to choose whether using sparse method
        self.nodes = num_nodes
        self.units = num_units

        adj = adj.to(device)

        if sparse:
            self.gat_1 = SpGraphAttentionLayer(in_features=2 * num_units, out_features=2*self.units, dropout=dropout,
                                             alpha=alpha)
            self.gat_2 = SpGraphAttentionLayer(in_features=2 * num_units, out_features=self.units, dropout=dropout,
                                               alpha=alpha)
        else:
            self.gat_1 = GraphAttentionLayer(in_features=2*num_units, out_features=2*self.units, dropout=dropout, alpha=alpha)
            self.gat_2 = GraphAttentionLayer(in_features=2 * num_units, out_features=self.units, dropout=dropout,
                                               alpha=alpha)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.A = Parameter(adj)
        #self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, X, state=None):
        #print('X shape:', X.shape)
        if state is None:
            state = X
        X1 = torch.cat((X, state), dim=2)
        #print('X1 shape:', X1.shape)
        graph_value = self.sigmoid(self.gat_1(X1, self.A))
        r, z = graph_value.chunk(2, dim=2)
        r_state = r * state
        X2 = torch.cat((X, r_state), dim=2)
        h_t1 = self.tan(self.gat_2(X2, self.A))
        new_h = z * state + (1 - z) * h_t1
        return new_h, new_h

# this GAT model obtained from paper:https://arxiv.org/abs/1710.10903
#
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout

        self.in_features = in_features

        self.out_features = out_features

        self.alpha = alpha

        #self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # h ->(b, n, f)
        b, n, f = input.shape
        input = input.view(b*n, f)
        h = torch.mm(input, self.W)

        h = h.view(b, n, self.out_features)
        N = n

        a_input = torch.cat([h.repeat(1, 1, N).view(b, N * N, -1), h.repeat(1, N, 1)], dim=2).view(b, N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=2)

        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.bmm(attention, h)

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        b, n, f = input.shape

        edge = adj.nonzero(as_tuple=False).t()
        #print('edge.shape:', edge.shape)
        h = torch.mm(input.view(b*n, f), self.W).view(b, n, -1)
        # h: N x out
        # using assert is to avoid error
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[:, edge[0, :], :], h[:, edge[1, :], :]), dim=2)
        # edge: 2*D x E
        #print('edge h shape:', edge_h.shape)
        edge_e = torch.exp(-self.leakyrelu(torch.matmul(edge_h, self.a).squeeze()))
        #print('edge e shape:', edge_e.shape)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        for i in range(b):
            if i == 0:
                e_rowsum = self.special_spmm(edge, edge_e[i], torch.Size([n, n]), torch.ones(size=(n, 1), device=dv))
                h_prime = self.special_spmm(edge, edge_e[i], torch.Size([n, n]), h[i])
            else:
                temp = self.special_spmm(edge, edge_e[i], torch.Size([n, n]), torch.ones(size=(n, 1), device=dv))
                e_rowsum = torch.cat([e_rowsum, temp], dim=0)
                #edge_e[i] = self.dropout(edge_e[i])
                temp = self.special_spmm(edge, edge_e[i], torch.Size([n, n]), h[i])
                h_prime = torch.cat([h_prime, temp], dim=0)

        e_rowsum = e_rowsum.view(b, n, -1)
        # e_rowsum: N x 1

        # edge_e: E
        h_prime = h_prime.view(b, n, -1)
        #print('h_prime:', h_prime.shape)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


