import os
import pickle as pk

import scipy.sparse as sp
from torch_geometric.data import Data

from utils import hermitian_feature


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