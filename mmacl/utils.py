import random
from itertools import permutations

import numpy as np
import torch
from torch import Tensor
import pandas as pd
from typing import List

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

def hyperedge_preprocess(hyperedge_index, num_nodes, num_edges):

        hyperedge_adj = pd.DataFrame(np.zeros((num_nodes, num_edges)), index=range(0, num_nodes), columns=range(0, num_edges))

        nodes = hyperedge_index.iloc[0, :].tolist()
        edges = hyperedge_index.iloc[1, :].tolist()

        for node, edge in zip(nodes, edges):
            hyperedge_adj[edge][node] = 1

        hyperedge_adj = torch.tensor(hyperedge_adj.values, dtype=torch.float32).to('cuda')

        return hyperedge_adj
    
def hypergraph_to_general_graph(hypergraph_adjacency):
    num_nodes = hypergraph_adjacency.shape[0]
    general_graph_adjacency = np.zeros((num_nodes, num_nodes))

    for hyperedge_index in range(hypergraph_adjacency.shape[1]):
        hyperedge = hypergraph_adjacency[:, hyperedge_index]
        nodes_in_hyperedge = np.where(hyperedge == 1)[0]

        for i in range(len(nodes_in_hyperedge)):
            for j in range(i + 1, len(nodes_in_hyperedge)):
                general_graph_adjacency[nodes_in_hyperedge[i], nodes_in_hyperedge[j]] = 1
                general_graph_adjacency[nodes_in_hyperedge[j], nodes_in_hyperedge[i]] = 1

    general_graph_adjacency = torch.tensor(general_graph_adjacency, dtype=torch.float32).to('cuda')
    return general_graph_adjacency

def create_graph2hyper_attention(graph_attention_matrix, hypergraph_matrix):
    graph_attention_matrix = graph_attention_matrix
    hypergraph_matrix = hypergraph_matrix
    hypergraph_attention_matrix = torch.zeros_like(hypergraph_matrix)

    for col_idx in range(hypergraph_matrix.size(1)):
        column = hypergraph_matrix[:, col_idx]
        non_zero_indices = torch.nonzero(column).squeeze()
        num_list = non_zero_indices.tolist()
        if isinstance(num_list, int):
            # num_combinations_list = list(num_list)
            hypergraph_attention_matrix[num_list, col_idx] = graph_attention_matrix[num_list][num_list].item()
            continue
        elif len(num_list) == 2:
            num_combinations_list = [(num_list[0], num_list[1]), (num_list[1], num_list[0])]
        else:
            num_combinations_list = list(permutations(num_list, 2))
        this_count = 0
        this_sum = 0
        for idx, combination in enumerate(num_combinations_list):
            from_node_number = combination[0]
            to_node_number = combination[1]
            if idx == 0:
                from_num_count = from_node_number
            if from_node_number != from_num_count:
                target_attention_score = this_sum/this_count
                hypergraph_attention_matrix[pre_from_node_number, col_idx] = target_attention_score
                this_count = 1
                this_sum = graph_attention_matrix[from_node_number][to_node_number].item()
                from_num_count = from_node_number
            else:
                this_sum += graph_attention_matrix[from_node_number][to_node_number].item()
                this_count += 1
                pre_from_node_number = combination[0]

            if idx == len(num_combinations_list) - 1:
                target_attention_score = this_sum/this_count
                hypergraph_attention_matrix[from_node_number, col_idx] = target_attention_score
    return hypergraph_attention_matrix
