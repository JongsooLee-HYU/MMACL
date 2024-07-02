import argparse
import random

import yaml
from tqdm import tqdm
import numpy as np
import torch

from mmacl.loader import DatasetLoader
from mmacl.models import HyperEncoder, MMACL
from mmacl.evaluation import linear_evaluation

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(num_negs):

    features, hyperedge_index = data.features, data.hyperedge_index
    num_nodes, num_edges = data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Encoder
    n_h, n_g = model(features, hyperedge_index, num_nodes, num_edges)
    n_h, n_g = model.node_projection(n_h), model.node_projection(n_g)
    loss_n = model.node_level_loss(n_h, n_g, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)

    loss = loss_n
    loss.backward()
    optimizer.step()
    return loss.item()



def node_classification_eval(num_splits=20):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    else:
        lr = 0.005
        max_epoch = 100
    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
    return accs 



if __name__ == '__main__':
    parser = argparse.ArgumentParser('MMACL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--proj_dim', type=int, default=1024)
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]

    data = DatasetLoader().load(args.dataset).to(args.device)

    params['dataset'] = args.dataset
    # params['hid_dim'] = args.hid_dim
    # params['proj_dim'] = args.proj_dim
    print(params)
    accs = []
    for seed in range(args.num_seeds):
        fix_seed(seed)
        encoder = HyperEncoder(args.alpha, data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = MMACL(encoder, params['proj_dim'], data.features.shape[1]).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss = train(num_negs=None)
        acc = node_classification_eval()

        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')

    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')
