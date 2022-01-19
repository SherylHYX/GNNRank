from typing import Union, Optional, Tuple

import numpy as np
import torch
from texttable import Texttable
import latextable
import scipy.sparse as sp
from scipy.stats import kendalltau, rankdata

from SpringRank import SpringRank
from comparison import syncRank_angle, syncRank, serialRank, btl, davidScore, eigenvectorCentrality, PageRank, rankCentrality, mvr
from comparison import SVD_RS, SVD_NRS

def set_grad(var):
    def hook(grad):
        print(grad.max(), grad.min())
    return hook


default_compare_names_all = ['DIGRAC']
default_metric_names = ['test kendall tau', 'test kendall p', 'val kendall tau', 'val kendall p', 'all kendall tau', 'all kendall p']
def print_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=True):
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
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if metric_names[i] in ['test kendall tau', 'val kendall tau', 'all kendall tau']:
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")

def print_overall_performance_mean_std(title:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               dataset_names:list=['animal'], print_latex:bool=True, print_std:bool=True):
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
        [len(dataset_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = title+'Data/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = dataset_names
    std = np.chararray(
        [len(dataset_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(dataset_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if title[:7] == 'kendall':
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=title +
                                    " performance.", label="table:"+title) + "\n")

def print_ablation_performance_mean_std(title:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               dataset_names:list=['animal'], print_latex:bool=True, print_std:bool=True):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.
    """
    split_ind = 3
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(dataset_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = title+'Data/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = dataset_names
    std = np.chararray(
        [len(dataset_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(dataset_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if title[:7] == 'kendall':
                best_values_left = -np.sort(-results_mean[i, :split_ind])[:2] # the bigger, the better
                best_values_right = -np.sort(-results_mean[i, split_ind:])[:2]
            else:
                best_values_left = np.sort(results_mean[i, :split_ind])[:2] # the smaller, the better
                best_values_right = np.sort(results_mean[i, split_ind:])[:2]
            
            ind_bool = results_mean[i]==best_values_left[0]
            ind_bool[split_ind:] = False
            final_res_show[i+1, 1:][ind_bool] = red_start + final_res_show[i+1, 1:][ind_bool] + both_end

            ind_bool = results_mean[i]==best_values_right[0]
            ind_bool[:split_ind] = False
            final_res_show[i+1, 1:][ind_bool] = red_start + final_res_show[i+1, 1:][ind_bool] + both_end
            if best_values_left[0] != best_values_left[1]:
                ind_bool = results_mean[i]==best_values_left[1]
                ind_bool[split_ind:] = False
                final_res_show[i+1, 1:][ind_bool] = blue_start + final_res_show[i+1, 1:][ind_bool] + both_end
            if best_values_right[0] != best_values_right[1]:
                ind_bool = results_mean[i]==best_values_right[1]
                ind_bool[:split_ind] = False
                final_res_show[i+1, 1:][ind_bool] = blue_start + final_res_show[i+1, 1:][ind_bool] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=title +
                                    " performance.", label="table:"+title) + "\n")

def print_and_analysis_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=True, 
                               metrics_of_interest: list=['upset_simple', 'upset_ratio'], methods_of_interest: list=['DIGRAC_plain_sort_emb_score']) -> list:
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
        metrics_of_interest: (list, optional) Metrics of interest to analyze.
        methods_of_interest: (list, optional) Methods of interest to analyze.

    Return:
        conclusions: (np.array) Conclusion array of the analysis, with size (len(methods_of_interest), len(metrics_of_interest)).
    
    Notation of the conlcusion:
    0: unknown; 
    1: best among all; 
    2: best compared with baselines only; 
    3: second best among all; 
    4: second best compared with baselines only
    5: 2&3
    6: none of above
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if metric_names[i] in ['test kendall tau', 'val kendall tau', 'all kendall tau']:
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")

    conclusions = np.zeros((len(methods_of_interest), len(metrics_of_interest)))
    for i, method_of_interest in enumerate(methods_of_interest):
        for j, metric_of_interest in enumerate(metrics_of_interest):
            try:
                curr = results_mean[metric_names.index(metric_of_interest)]
                curr_of_interest = curr[compare_names_all.index(method_of_interest)]
                if 'mvr' in compare_names_all:
                    baseline_start_ind = compare_names_all.index('mvr')
                else:
                    baseline_start_ind = compare_names_all.index('SpringRank')
                baselines_plus_interest = np.concatenate((curr[baseline_start_ind:], curr_of_interest.reshape((1,))))
                if metric_of_interest not in ['upset', 'upset_ratio', 'upset_simple', 'test kendall p', 'val kendall p', 'all kendall p']:
                    best_values = -np.sort(-curr)[:2] # the bigger, the better
                    best_values_baselines = -np.sort(-baselines_plus_interest)[:2]
                else:
                    best_values = np.sort(curr)[:2] # the smaller, the better
                    best_values_baselines = np.sort(baselines_plus_interest)[:2]
                if curr_of_interest == best_values[0]:
                    conclusion = 1
                elif curr_of_interest == best_values_baselines[1]:
                    conclusion = 4
                elif curr_of_interest == best_values[1]:
                    if curr_of_interest == best_values_baselines[0]:
                        conclusion = 5
                    else:
                        conclusion = 3
                elif curr_of_interest == best_values_baselines[0]:
                    conclusion = 2
                else:
                    conclusion = 6
            except ValueError:
                conclusion = 0
            conclusions[i, j] = conclusion
    return conclusions

class Prob_Imbalance_Loss(torch.nn.Module):
    r"""An implementation of the probablistic imbalance loss function.
    Args:
        F (int or NumPy array, optional): Number of pairwise imbalance socres to consider, or the meta-graph adjacency matrix.
    """

    def __init__(self, F: Optional[Union[int, np.ndarray]] = None):
        super(Prob_Imbalance_Loss, self).__init__()
        if isinstance(F, int):
            self.sel = F
        elif F is not None:
            K = F.shape[0]
            self.sel = 0
            for i in range(K-1):
                for j in range(i+1, K):
                    if (F[i, j] + F[j, i]) > 0:
                        self.sel += 1

    def forward(self, P: torch.FloatTensor, A: Union[torch.FloatTensor, torch.sparse_coo_tensor], 
    K: int, normalization: str = 'plain', threshold: str = 'naive') -> torch.FloatTensor:
        """Making a forward pass of the probablistic imbalance loss function.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
            A: (PyTorch FloatTensor, can be sparse) Adjacency matrix A
            K: (int) Number of clusters
            normalization: (str, optional) normalization method:
                'vol_sum': Normalized by the sum of volumes, the default choice.
                'vol_max': Normalized by the maximum of volumes.            
                'vol_min': Normalized by the minimum of volumes.   
                'plain': No normalization, just CI.   
            threshold: (str, optional) normalization method:
                'sort': Picking the top beta imbalnace values, the default choice.
                'std': Picking only the terms 3 standard deviation away from null hypothesis.             
                'naive': No thresholding, suming up all K*(K-1)/2 terms of imbalance values.  
        Returns:
            loss value, roughly in [0,1].
        """
        assert normalization in ['vol_sum', 'vol_min', 'vol_max',
                                 'plain'], 'Please input the correct normalization method name!'
        assert threshold in [
            'sort', 'std', 'naive'], 'Please input the correct threshold method name!'

        device = A.device
        # for numerical stability
        epsilon = torch.FloatTensor([1e-10]).to(device)
        # first calculate the probabilitis volumns for each cluster
        vol = torch.zeros(K).to(device)
        for k in range(K):
            vol[k] = torch.sum(torch.matmul(
                A + torch.transpose(A, 0, 1), P[:, k:k+1]))
        second_max_vol = torch.topk(vol, 2).values[1] + epsilon
        result = torch.zeros(1).to(device)
        imbalance = []
        if threshold == 'std':
            imbalance_std = []
        for k in range(K-1):
            for l in range(k+1, K):
                w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l]))
                w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k]))
                if (w_kl-w_lk).item() != 0:
                    if threshold != 'std' or np.power((w_kl-w_lk).item(), 2)-9*(w_kl+w_lk).item() > 0:
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk + epsilon)
                        imbalance.append(curr)
                    else:  # below-threshold values in the 'std' thresholding scheme
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk + epsilon)
                        imbalance_std.append(curr)
        imbalance_values = [curr.item() for curr in imbalance]
        if threshold == 'sort':
            # descending order
            ind_sorted = np.argsort(-np.array(imbalance_values))
            for ind in ind_sorted[:int(self.sel)]:
                result += imbalance[ind]
            # take negation to be minimized
            return torch.ones(1, requires_grad=True).to(device) - result/self.sel
        elif len(imbalance) > 0:
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance))
        elif threshold == 'std':  # sel is 0, then disregard thresholding
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance_std))
        else:  # nothing has positive imbalance
            return torch.ones(1, requires_grad=True).to(device)

def get_imbalance_distribution_and_flow(labels: Union[list, np.array, torch.LongTensor],
                                        num_clusters: int,
                                        A: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                                        F: Optional[Union[int, np.ndarray]] = None,
                                        normalizations: list = ['plain'],
                                        thresholds: list = ['naive']) -> Tuple[list, np.array, np.array]:
    r"""Computes imbalance values, distribution of labels, 
    and predicted meta-graph flow matrix.

    Args:
        labels: (list, np.array, or torch.LongTensor) Predicted labels.
        num_clusters: (int) Number of clusters.
        A: (torch.FloatTensor or torch.sparse_coo_tensor) Adjacency matrix.
        F: (int or  np.ndarray, optional) Number of selections in "sort" flavor or 
            the meta-graph adjacency matrix.
        normalizations: (list) Normalization methods to consider, 
            default is ['plain'].
        thresholds: (list) Thresholding methods to consider, 
            default is ['naive'].

    :rtype: 
        imbalance_list: (list) List of imbalance values from different loss functions.
        labels_distribution: (np.array) Array of distribution of labels.
        flow_mat: (np.ndarray) Predicted meta-graph flow matrix.
    """
    P = torch.zeros(labels.shape[0], num_clusters).to(A.device)
    loss = Prob_Imbalance_Loss(F)
    for k in range(num_clusters):
        P[labels == k, k] = 1
    labels_distribution = np.array(P.sum(0).to('cpu').numpy(), dtype=int)
    imbalance_list = []
    for threshold in thresholds:
        for normalization in normalizations:
            imbalance_list.append(
                1-loss(P, A, num_clusters, normalization, threshold).item())
    flow_mat = np.ones([num_clusters, num_clusters])*0.5
    for k in range(num_clusters-1):
        for l in range(k+1, num_clusters):
            w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l])).item()
            w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k])).item()
            if (w_kl + w_lk) > 0:
                flow_mat[k, l] = w_kl/(w_kl + w_lk)
                flow_mat[l, k] = w_lk/(w_kl + w_lk)

    return imbalance_list, labels_distribution, flow_mat


def obtain_ranking_from_clusters(labels: Union[list, np.array, torch.LongTensor],
num_clusters: int, A: Union[torch.FloatTensor, torch.sparse_coo_tensor], model_name: str='SpringRank') -> np.array:
    r"""Assign rankings (with ties) to cluster assignments by fitting predicted meta-graph flow matrix.

    Args:
        labels: (list, np.array, or torch.LongTensor) Predicted labels.
        num_clusters: (int) Number of clusters.
        A: (torch.FloatTensor or torch.sparse_coo_tensor) Adjacency matrix.
        model_name: (str) Baseline model for ranking that should be applied to the fitted meta-graph.

    :rtype: 
        ranking: (np.array) Updated rankings.
        """
    P = torch.zeros(labels.shape[0], num_clusters).to(A.device)
    for k in range(num_clusters):
        P[labels == k, k] = 1
    ranking = np.zeros(A.shape[0])
    scores = np.zeros((A.shape[0], 1))
    if len(list(set(labels))) == 1:
        return torch.LongTensor(labels).float().numpy(), scores

    flow_mat = np.ones([num_clusters, num_clusters])*0.5
    for k in range(num_clusters-1):
        for l in range(k+1, num_clusters):
            w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l])).item()
            w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k])).item()
            if (w_kl + w_lk) > 0:
                flow_mat[k, l] = w_kl/(w_kl + w_lk)
                flow_mat[l, k] = w_lk/(w_kl + w_lk)
    flow_mat_torch = torch.FloatTensor(flow_mat).to(A.device)
    flow_mat = sp.csr_matrix(flow_mat)
    try:
        label_mapping = syncRank(sp.csr_matrix(flow_mat))
        if model_name == 'SpringRank':
            label_mapping_scores = SpringRank(flow_mat,alpha=0,l0=1,l1=1)
        elif model_name == 'serialRank':
            label_mapping_scores = serialRank(flow_mat)
        elif model_name == 'btl':
            label_mapping_scores = btl(flow_mat)
        elif model_name == 'davidScore':
            label_mapping_scores = davidScore(flow_mat)
        elif model_name == 'eigenvectorCentrality':
            label_mapping_scores = eigenvectorCentrality(flow_mat)
        elif model_name == 'PageRank':
            label_mapping_scores = PageRank(flow_mat)
        elif model_name == 'rankCentrality':
            label_mapping_scores = rankCentrality(flow_mat)
        elif model_name == 'syncRank':
            label_mapping = syncRank(flow_mat)
            label_mapping_scores = syncRank_angle(flow_mat) # scores
        elif model_name == 'mvr':
            label_mapping = mvr(flow_mat)
        elif model_name == 'SVD_RS':
            label_mapping_scores = SVD_RS(flow_mat)
        elif model_name == 'SVD_NRS':
            label_mapping_scores = SVD_NRS(flow_mat)
        else:
            raise NameError('Please input the correct model name from:\
                SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                PageRank, rankCentrality, mvr, SVD_RS, SVD_NRS, instead of {}!'.format(model_name))
        if model_name not in ['mvr']:
            label_mapping_scores_torch = torch.FloatTensor(label_mapping_scores.reshape(label_mapping_scores.shape[0], 1)).to(A.device)
            if label_mapping_scores.min() < 0:
                label_mapping_scores_torch = torch.sigmoid(label_mapping_scores_torch)
            upset1 = calculate_upsets(flow_mat_torch, label_mapping_scores_torch)
            upset2 = calculate_upsets(flow_mat_torch, -label_mapping_scores_torch)
            
            if model_name not in ['syncRank']:
                if upset1.detach().item() > upset2.detach().item():
                    label_mapping_scores = -label_mapping_scores
                label_mapping = rankdata(-label_mapping_scores, 'min')
            else:
                if upset1.detach().item() > upset2.detach().item():
                    label_mapping = 1 + label_mapping.max()-label_mapping
    except Exception:
        # print('TypeError encountered with predicted flow matrix: {}'.format(flow_mat))
        return torch.LongTensor(labels).float().numpy(), scores
    for k in range(num_clusters):
        ranking[labels==k] = label_mapping[k]
        scores[labels==k] = label_mapping_scores[k]
    return ranking, scores

def calculate_upsets(A: torch.FloatTensor,
                     score: torch.FloatTensor, style: str='ratio', margin: float=0.01)-> torch.FloatTensor:
    r"""Calculate upsets from rankings (with ties). 
    Convention: r_i (the score for the i-th node) larger, better skill, smaller ranks, larger out-degree.

    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Ranking scores, with shape (num_nodes, 1).
        style: (str, optional) Styles of loss to choose, default ratio.
        margain: (float, optional) Margin for which we need to hold for the margin version, default 0.01.
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        """
    # for numerical stability
    epsilon = torch.FloatTensor([1e-8]).to(score)
    # e = torch.ones_like(score)
    M = A - torch.transpose(A, 0, 1)
    indices = (M != 0)
    T1 = score - score.T # torch.mm(score, torch.transpose(e, 0, 1)) - torch.mm(e, torch.transpose(score, 0, 1))
    if style == 'simple':
        upset = torch.mean(torch.pow(torch.sign(T1[indices]) - torch.sign(M[indices]), 2))
    elif style == 'margin':
        upset = torch.mean((M + torch.abs(M)).multiply(torch.nn.ReLU()(-T1 + margin))[indices]) # margin loss
    elif style == 'naive':
        upset = torch.sum(torch.sign(T1[indices]) != torch.sign(M[indices]))/torch.sum(indices)
    else: # 'ratio'
        T2 = score + score.T + epsilon  # torch.mm(score, torch.transpose(e, 0, 1)) + torch.mm(e, torch.transpose(score, 0, 1)) + epsilon
        T = torch.div(T1, T2) # torch.nan_to_num(torch.div(T1, T2))
        M2 = A + A.T + epsilon
        M3 = torch.div(M, M2) # torch.nan_to_num(torch.div(M, M2))
        powers = torch.pow((M3-T)[indices], 2)
        # print(torch.max(T2), torch.min(T2))
        upset = torch.mean(powers)# /M.numel()
        # the value is from 0 to 2, as the difference is from 0 to 2 (1 and -1)
        # print(torch.sum(indices), torch.max(torch.pow((M3-T)[indices], 2)), torch.max(M3), torch.max(T))
        
    # upset1 = torch.sum(torch.sign(T1) != torch.sign(M))
    # upset2 = torch.sum(torch.sign(-T1) != torch.sign(M))
    # upset = torch.min(upset1, upset2)/M.numel()
    # upset.register_hook(set_grad(upset))
    return upset