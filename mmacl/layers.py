from typing import Optional, Callable

from .utils import create_graph2hyper_attention, hypergraph_to_general_graph, hyperedge_preprocess
from itertools import permutations
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import numpy as np
import pandas as pd

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

class ScaledDotProductAttention_hyper(nn.Module):
    ''' Scaled Dot-Product Attention for Hypergraph'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, v, orginal_attention, feature, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        if feature != True:
            orginal_attention = orginal_attention.permute(0,2,1)
        orginal_attention = F.softmax(orginal_attention, dim=-1)
        attn =  F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        attn = torch.add(orginal_attention, attn)

        output = torch.matmul(attn, v)

        return output, attn
    
class ProposedConv(nn.Module):

    def __init__(self, transfer, alpha, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.2, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.in_features = in_dim
        self.concat = True
        self.edge = False
        self.hid_dim = hid_dim
        self.out_features = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm
        self.alpha = alpha

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)


        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        self.weight_g = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.a_g = nn.Parameter(torch.Tensor(2*self.out_features, 1))

        self.leakyrelu_g = nn.LeakyReLU(self.alpha)

        self.word_context = nn.Embedding(1, self.out_features)
      
        self.attention1 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout = self.dropout)
        self.attention2 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout = self.dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        self.weight_g.data.uniform_(-stdv, stdv)
        self.a_g.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        residual = x
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        hyperedge_index = pd.DataFrame(hyperedge_index.cpu().numpy())
        hyperedge_adj = hyperedge_preprocess(hyperedge_index, num_nodes, num_edges)

        # general part -start
        connected_general_graph_adjacency_matrix = hypergraph_to_general_graph(hyperedge_adj.cpu().numpy())
        connected_general_graph_adjacency_matrix.fill_diagonal_(1)
        
        Wh_g = torch.mm(x, self.weight_g) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e_g = self._prepare_attentional_mechanism_input(Wh_g)

        zero_vec_g = -9e15*torch.ones_like(e_g)
        attention_g = torch.where(connected_general_graph_adjacency_matrix > 0, e_g, zero_vec_g)

        attention_g = F.dropout(attention_g, self.dropout, training=self.training)
        attention_g = F.softmax(attention_g, dim=1)

        graph2hyper = create_graph2hyper_attention(attention_g, hyperedge_adj)
        graph2hyper = graph2hyper.unsqueeze(0).permute(0,2,1)
            
        x = x.unsqueeze(0)
        adj = hyperedge_adj.unsqueeze(0).permute(0,2,1)

        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias        

        N1 = adj.shape[1] #number of edge
        N2 = adj.shape[2] #number of node


        q1 = self.word_context.weight[0:].view(1, 1, -1).repeat(x.shape[0], N1, 1).view(x.shape[0], N1, self.out_features)

        edge, att1 = self.attention1(q1, x_4att, x, graph2hyper, feature = True, mask = adj)
        
        att2he = att1.permute(0,2,1) 
        edge_4att = edge.matmul(self.weight3)

      
        node, att2hn = self.attention2(x_4att, edge_4att, edge, graph2hyper, feature = False, mask = adj.transpose(1, 2))

        att2_he_out = torch.mm(att2he.squeeze(0),att2he.squeeze(0).t())
        att2_hn_out = torch.mm(att2hn.squeeze(0),att2hn.squeeze(0).t())

        attention_out = torch.add(attention_g, torch.add(att2_he_out, att2_hn_out))
        h_prime_g = torch.matmul(attention_out, Wh_g)

        if self.concat:
            h_prime_g = F.elu(h_prime_g)
        else:
            h_prime_g

        if self.concat:
            node = F.relu(node)
            edge = F.relu(edge)

        node = node.squeeze()

        return node, h_prime_g

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a_g[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a_g[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu_g(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
