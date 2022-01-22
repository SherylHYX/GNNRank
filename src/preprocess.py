# standard libaries
import os
import random
import pickle as pk

# third-party libraries
import torch
import scipy.sparse as sp
import numpy.random as rnd

# internel
from utils import  DSBM, ERO
from extract_network import extract_network
from generate_data import to_dataset, to_dataset_no_label, to_dataset_no_split



def load_data_from_memory(root, name=None):
    data = pk.load(open(root, 'rb'))
    if os.path.isdir(root) == False:
        try:
            os.makedirs(root)
        except FileExistsError:
            pass
    return [data]

def load_real_data(dataset):
    A = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '../data/'+dataset+'adj.npz'))
    return A

def load_data(args, random_seed):
    rnd.seed(random_seed)
    random.seed(random_seed)
    label = None
    train_mask = None
    val_mask = None
    test_mask = None
    default_name_base =  'trials' + str(args.num_trials) + 'train_r' + str(int(100*args.train_ratio)) + 'test_r' + str(int(100*args.test_ratio))
    if args.dataset[:4] == 'DSBM' or args.dataset[:3] == 'ERO':
        default_name_base += 'seed' + str(random_seed)
    save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/'+args.dataset+default_name_base+'.pk')
    if (not args.regenerate_data) and os.path.exists(save_path):
        print('Loading existing data!')
        data = load_data_from_memory(save_path, name=None)[0]
    else:
        print('Generating new data or new data splits!')
        if args.dataset[:4] == 'DSBM':
            if args.sp_style == 'random':
                A, label = DSBM(N=args.N, K=args.K, p=args.p,
                                F=args.F_data, size_ratio=args.size_ratio)
            else:
                raise NameError(
                    'Please input the correct sparsity option! Only "random" is supported for now.')
            A, label = extract_network(A, label)
            data = to_dataset(args, A, label, save_path=save_path,
                          load_only=args.load_only)
        elif args.dataset[:3] == 'ERO':
            A, label = ERO(n=args.N, p=args.p, eta=args.eta, style=args.ERO_style)
            A, label = extract_network(A, label)
            data = to_dataset_no_split(A, args.K, torch.LongTensor(label), save_path=save_path,
                          load_only=args.load_only)
        else:
            A = load_real_data(args.dataset)
            data = to_dataset_no_label(A, args.K, save_path=save_path,
                          load_only=args.load_only)


    if data.y is not None:
        label = data.y.data.numpy().astype('int')
    if hasattr(data, 'train_mask'):
        train_mask = data.train_mask.data.numpy().astype('bool_')
        val_mask = data.val_mask.data.numpy().astype('bool_')
        test_mask = data.test_mask.data.numpy().astype('bool_')

    return label, train_mask, val_mask, test_mask, data.x, data.A

