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
    

if __name__ == "__main__":
    preprocess_hypergcn_dataset('coauthorship', 'cora')
    preprocess_hypergcn_dataset('coauthorship', 'dblp')
    preprocess_hypergcn_dataset('cocitation', 'cora')
    preprocess_hypergcn_dataset('cocitation', 'citeseer')
    preprocess_hypergcn_dataset('cocitation', 'pubmed')
