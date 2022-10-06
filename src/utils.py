import csv

import numpy as np
import scipy.sparse as sp
import torch
from texttable import Texttable
import latextable
from sklearn.preprocessing import normalize, StandardScaler
from scipy.stats import rankdata


def write_log(args, path):
    with open(path+'/settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for para in args:
            writer.writerow([para, args[para]])
    return

def ERO(n: int, p: float, eta: float, style:str='uniform') -> sp.csr_matrix:
    """A Erdos-Renyi Outliers (ERO) model graph generator.
    Args:
        n: (int) Number of nodes.
        p: (float) Sparsity value, edge probability.
        eta : (float) Noise level, between 0 and 1.
        style: (string) How to generate ratings:
            'uniform': Uniform.
            'gamma': Gamma distribution with shape 0.5 and scale 1.

    Returns:
        R: (sp.csr_matrix) a sparse n by n matrix of pairwise comparisons.
        labels: (np.array) ground-truth ranking.
    """
    if style == 'uniform':
        scores = np.random.rand(n, 1)
        R_noise = np.random.rand(n, n) * 2 - 1
    elif style == 'gamma':
        scores = np.random.gamma(shape=0.5, scale=1, size=(n, 1))
        R_noise = np.random.rand(n, n) * 4 - 2 # 0.95 percentile for gamma(0.5, 1) is about 1.9207
    labels = rankdata(-scores, 'min')
    R_GT = scores - scores.transpose() # use broadcasting
    R_choice = np.random.rand(n, n)
    R = np.zeros((n, n))
    R[R_choice<=p] = R_noise[R_choice<=p]
    R[R_choice<=p*(1-eta)] = R_GT[R_choice<=p*(1-eta)]
    lower_ind = np.tril_indices(n)
    diag_ind = np.diag_indices(n)
    R[lower_ind] = -R.transpose()[lower_ind]
    R[diag_ind] = 0
    R[R<0] = 0
    return sp.csr_matrix(R), labels

def get_powers_sparse(A, hop=3, tau=0.1):
    '''
    function to get adjacency matrix powers
    inputs:
    A: directed adjacency matrix
    hop: the number of hops that would like to be considered for A to have powers.
    tau: the regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I, 
        where I is the identity matrix. If tau=0, then we have no self-loops to add.
    output: (torch sparse tensors)
    A_powers: a list of A powers from 0 to hop
    '''
    A_powers = []

    shaping = A.shape
    adj0 = sp.eye(shaping[0])

    A_bar = normalize(A+tau*adj0, norm='l1')  # l1 row normalization
    tmp = A_bar.copy()
    adj0_new = sp.csc_matrix(adj0)
    ind_power = A.nonzero()
    A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
        adj0_new.nonzero()), torch.FloatTensor(adj0_new.data), shaping))
    A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
        ind_power), torch.FloatTensor(np.array(tmp[ind_power]).flatten()), shaping))
    if hop > 1:
        A_power = A.copy()
        for _ in range(2, int(hop)+1):
            tmp = tmp.dot(A_bar)  # get A_bar powers
            A_power = A_power.dot(A)
            ind_power = A_power.nonzero()  # get indices for M matrix
            tmp = tmp.dot(A_bar)  # get A_bar powers
            A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(
                ind_power), torch.FloatTensor(np.array(tmp[ind_power]).flatten()), shaping))

            # A_powers.append(torch.sparse_coo_tensor(torch.LongTensor(tmp.nonzero()), torch.FloatTensor(tmp.data), shaping))
    return A_powers



def hermitian_feature(A, num_clusters):
    """ create Hermitian feature  (rw normalized)
    inputs:
    A : adjacency matrix
    num_clusters : number of clusters

    outputs: 
    features_SVD : a feature matrix from SVD of Hermitian matrix
    """
    H = (A-A.transpose()) * 1j
    H_abs = np.abs(H)  # (np.real(H).power(2) + np.imag(H).power(2)).power(0.5)
    D_abs_inv = sp.diags(1/np.array(H_abs.sum(1))[:, 0])
    H_rw = D_abs_inv.dot(H)
    u, _, _ = sp.linalg.svds(H_rw, k=num_clusters)
    features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
    scaler = StandardScaler().fit(features_SVD)
    features_SVD = scaler.transform(features_SVD)
    return features_SVD


def scipy_sparse_to_torch_sparse(A):
    A = sp.csr_matrix(A)
    return torch.sparse_coo_tensor(torch.LongTensor(A.nonzero()), torch.FloatTensor(A.data), A.shape)

default_compare_names_all = ['MLP', 'GCN']
default_metric_names = ['test acc', 'test auc', 'test F1', 'val acc', 'val auc','val F1', 'all acc', 'all auc','all F1']
def print_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=False):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=50)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(results.std(0),2))
    results_mean = np.transpose(np.round(results.mean(0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    else:
        plus_minus = np.chararray(
            [len(metric_names)-2, len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:-2, 1:] = final_res_show[1:-2, 1:] + plus_minus + std[:-2]
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            best_values = -np.sort(-results_mean[i])[:2]
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")
