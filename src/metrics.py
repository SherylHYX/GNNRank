import numpy as np
import torch
from texttable import Texttable
import latextable


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
    M = A - torch.transpose(A, 0, 1)
    indices = (M != 0)
    T1 = score - score.T
    if style == 'simple':
        upset = torch.mean(torch.pow(torch.sign(T1[indices]) - torch.sign(M[indices]), 2))
    elif style == 'margin':
        upset = torch.mean((M + torch.abs(M)).multiply(torch.nn.ReLU()(-T1 + margin))[indices]) # margin loss
    elif style == 'naive':
        upset = torch.sum(torch.sign(T1[indices]) != torch.sign(M[indices]))/torch.sum(indices)
    else: # 'ratio'
        T2 = score + score.T + epsilon
        T = torch.div(T1, T2)
        M2 = A + A.T + epsilon
        M3 = torch.div(M, M2)
        powers = torch.pow((M3-T)[indices], 2)
        upset = torch.mean(powers)# /M.numel()
        # the value is from 0 to 2, as the difference is from 0 to 2 (1 and -1)

    return upset