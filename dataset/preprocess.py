import os.path as osp
import pickle
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp


def load_hypergcn_dataset(type, name):
    dataset_dir = osp.join('dataset','HyperGCN', type, name)
    with open(osp.join(dataset_dir, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
    with open(osp.join(dataset_dir, 'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)
    with open(osp.join(dataset_dir, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    return features, hypergraph, labels


def save_preprocessed_dataset(type, name, features, hypergraph, labels):
    if type == 'etc':
        new_dir = osp.join('dataset', name)
    else:
        new_dir = osp.join('dataset', type, name)
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    with open(osp.join(new_dir, 'features.pickle'), 'wb') as f:
        pickle.dump(features, f)
    with open(osp.join(new_dir, 'hypergraph.pickle'), 'wb') as f:
        pickle.dump(hypergraph, f)
    with open(osp.join(new_dir, 'labels.pickle'), 'wb') as f:
        pickle.dump(labels, f)

    
def save_masks(type, name, index, masks):
    if type == 'etc':
        new_dir = osp.join('dataset', name, 'splits')
    else:
        new_dir = osp.join('dataset', type, name, 'splits')
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    with open(osp.join(new_dir, f'{index}.pickle'), 'wb') as f:
        pickle.dump(masks, f)


def generate_random_split(num_nodes, train_ratio, val_ratio, seed):
    # Random split for node classification
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    generator = np.random.default_rng(seed)
    idx_randperm = generator.permutation(num_nodes)
    train_mask = np.full(num_nodes, False)
    val_mask = np.full(num_nodes, False)
    test_mask = np.full(num_nodes, False)

    train_mask[idx_randperm[:num_train]] = True
    val_mask[idx_randperm[num_train:num_train + num_val]] = True
    test_mask[idx_randperm[num_train + num_val:]] = True

    return {
        'train_mask': train_mask, 
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def preprocess_hypergcn_dataset(type, name):
    features, hypergraph, labels = load_hypergcn_dataset(type, name)

    edge_set = set(hypergraph.keys())
    node_set = set()
    for edge in edge_set:
        nodes = hypergraph[edge]
        node_set.update(nodes)
        
    node_idx = list(node_set)
    node_to_num = {}
    num = 0
    for node in node_set:
        node_to_num[node] = num
        num += 1

    for edge in edge_set:
        nodes = hypergraph[edge]
        new_nodes = []
        for node in nodes:
            new_nodes.append(node_to_num[node])
        hypergraph[edge] = new_nodes
    
    features = features[node_idx] 
    labels = list(np.array(labels)[node_idx])

    save_preprocessed_dataset(type, name, features, hypergraph, labels)

    num_nodes = features.shape[0]
    for i in range(20):
        masks = generate_random_split(num_nodes, 0.1, 0.1, seed=i)
        save_masks(type, name, i, masks)
    print(f'Finish preprocess hypergcn dataset: {type}, {name}')


def preprocess_other_dataset(name):
    file_name = f'{name}.content'
    p2idx_features_labels = osp.join('dataset', name, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{name}.edges'
    p2edges_unordered = osp.join('dataset', name, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1

    ############################################################################
    features = features[:num_nodes]
    labels = labels[:num_nodes]
    
    if name in ['zoo', 'ModelNet40']:
        labels -= 1

    edge_set = set(edge_index[1])
    node_set = set()
    for edge in edge_set:
        nodes = edge_index[0, edge_index[1] == edge].tolist()
        node_set.update(nodes)
        
    node_idx = list(node_set)
    node_to_num = {}
    num = 0
    for node in node_set:
        node_to_num[node] = num
        num += 1

    for edge in edge_set:
        nodes = edge_index[0, edge_index[1] == edge].tolist()
        new_nodes = []
        for node in nodes:
            new_nodes.append(node_to_num[node])
        edge_index[0, edge_index[1] == edge] = new_nodes
    
    features = features[node_idx] 
    labels = labels[node_idx].tolist()
    hyperedge_index = edge_index
    hyperedge_index[1] -= num_nodes

    hypergraph = {}
    for edge in edge_set:
        nodes = edge_index[0, edge_index[1] == (edge - num_nodes)].tolist()
        hypergraph[edge - num_nodes] = nodes

    save_preprocessed_dataset('etc', name, features, hypergraph, labels)

    num_nodes = features.shape[0]
    for i in range(20):
        masks = generate_random_split(num_nodes, 0.1, 0.1, seed=i)
        save_masks('etc', name, i, masks)
    print(f'Finish preprocess other dataset: {name}')
    

if __name__ == "__main__":
    preprocess_hypergcn_dataset('coauthorship', 'cora')
    preprocess_hypergcn_dataset('coauthorship', 'dblp')
    preprocess_hypergcn_dataset('cocitation', 'cora')
    preprocess_hypergcn_dataset('cocitation', 'citeseer')
    preprocess_hypergcn_dataset('cocitation', 'pubmed')

    preprocess_other_dataset('zoo')
    preprocess_other_dataset('20newsW100')
    preprocess_other_dataset('Mushroom')
    preprocess_other_dataset('NTU2012')
    preprocess_other_dataset('ModelNet40')