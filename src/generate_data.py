import os
import random
import math
import pickle as pk

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import numpy.random as rnd

from param_parser import parameter_parser
from extract_network import extract_network
from utils import DSBM, hermitian_feature

def to_dataset(args, A, label, save_path, load_only=False, features=None):
    labels = label
    N = A.shape[0]
    idx = np.arange(N)
    num_clusters =  int(np.max(labels) + 1)
    clusters_sizes = [int(sum(labels==i)) for i in range(num_clusters)]
    test_sizes = [math.ceil(clusters_sizes[i] * args.test_ratio) for i in range(num_clusters)]
    val_ratio = 1 - args.train_ratio - args.test_ratio
    val_sizes = [math.ceil(clusters_sizes[i] * val_ratio) for i in range(num_clusters)]
    
    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [], []
    for _ in range(args.num_trials):
        idx_test = []
        idx_val = []
        for i in range(num_clusters):
            idx_test_ind = random.sample(range(clusters_sizes[i]), k=test_sizes[i])
            idx_test.extend((np.array(idx)[labels==i])[idx_test_ind])
        idx_remain = list(set(idx).difference(set(idx_test))) # the rest of the indices
        clusters_sizes_remain = [int(sum(labels[idx_remain]==i)) for i in range(num_clusters)]
        for i in range(num_clusters):
            idx_val_ind = random.sample(range(clusters_sizes_remain[i]), k=val_sizes[i])
            idx_val.extend((np.array(idx_remain)[labels[idx_remain]==i])[idx_val_ind])
        idx_train = list(set(idx_remain).difference(set(idx_val))) # the rest of the indices

        train_indices = idx_train
        val_indices = idx_val
        test_indices = idx_test

        train_mask = np.zeros((labels.shape[0], 1), dtype=int)
        train_mask[train_indices, 0] = 1
        train_mask = np.squeeze(train_mask, 1)
        val_mask = np.zeros((labels.shape[0], 1), dtype=int)
        val_mask[val_indices, 0] = 1
        val_mask = np.squeeze(val_mask, 1)
        test_mask = np.zeros((labels.shape[0], 1), dtype=int)
        test_mask[test_indices, 0] = 1
        test_mask = np.squeeze(test_mask, 1)

        mask = {}
        mask['train'] = train_mask
        mask['val'] = val_mask
        mask['test'] = test_mask
        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()
    
        masks['train'].append(mask['train'].unsqueeze(-1))
        masks['val'].append(mask['val'].unsqueeze(-1))
        masks['test'].append(mask['test'].unsqueeze(-1))
    
    label = label - np.amin(label)
    num_clusters = np.amax(label)+1
    label = torch.LongTensor(label)

    s_A = sp.csr_matrix(A)
    indices = torch.LongTensor(s_A.nonzero())
    
    if features is None:
        features = hermitian_feature(A, num_clusters)

    data = Data(x=features, edge_index=indices, edge_weight=None, y=label,A=s_A)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)
    if not load_only:
        if os.path.isdir(os.path.dirname(save_path)) == False:
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for best {}!'.format(os.path.dirname(save_path)))
        pk.dump(data, open(save_path, 'wb'))
    return data

def to_dataset_no_label(A, num_clusters, save_path, load_only=False, features=None):
    if features is None:
        features = hermitian_feature(A, num_clusters)

    data = Data(x=features, y=None,A=sp.csr_matrix(A))
    if not load_only:
        if os.path.isdir(os.path.dirname(save_path)) == False:
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for best {}!'.format(os.path.dirname(save_path)))
        pk.dump(data, open(save_path, 'wb'))
    return data

def to_dataset_no_split(A, num_clusters, label, save_path, load_only=False, features=None):
    if features is None:
        features = hermitian_feature(A, num_clusters)

    data = Data(x=features, y=label,A=sp.csr_matrix(A))
    if not load_only:
        if os.path.isdir(os.path.dirname(save_path)) == False:
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for best {}!'.format(os.path.dirname(save_path)))
        pk.dump(data, open(save_path, 'wb'))
    return data


def main():
    args = parameter_parser()
    rnd.seed(args.seed)
    random.seed(args.seed)
    if args.sp_style == 'random':
        A, label = DSBM(N=args.N, K=args.K, p=args.p, F = args.F_data, size_ratio=args.size_ratio)
    else:
        raise NameError('Please input the correct sparsity option: currently only "random".')
    A, label = extract_network(A,label)
    default_values = [args.p,args.fill_val, args.K,args.N, args.train_ratio, args.test_ratio,args.size_ratio,args.ambient,args.eta, args.num_trials]
    default_name_base = '_'.join([str(int(100*value)) for value in default_values])+ '_' + args.F_style+ '_' + args.sp_style
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/DSBM/'+default_name_base+'.pk')
    _ = to_dataset(args,A, label, save_path = save_path)
    return

if __name__ == "__main__":
    main()