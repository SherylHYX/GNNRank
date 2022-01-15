import argparse
import os

import torch

from utils import meta_graph_generation

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Ranking.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--debug', '-D',action='store_true', default=False,
                        help='Debugging mode, minimal setting.')
    parser.add_argument('--seed', type=int, default=31, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, #default = 0.01
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='(Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='(Initial) learning rate for proximal gradient step.')
    parser.add_argument('--trainable_alpha',action='store_true', default=False,
                        help='Whether to set the proximal gradient step learning rate to be trainable.')
    parser.add_argument('--Fiedler_layer_num', type=int, default=5,
                        help='The number of proximal gradient steps in calculating the Fiedler vector.')
    parser.add_argument('--train_with', type=str, default='anchor_dist',
                        help='To train GNNs with anchor_dist, anchor_innerproduct, emb_score or emb_baseline.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use. Adam or SGD in our case.')
    parser.add_argument('--pretrain_with', type=str, default='dist',
                        help='Variant to pretrain with, dist, innerproduct or serial_similarity.')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of epochs to pretrain.')
    parser.add_argument('--cluster_rank_baseline', type=str, default='SpringRank',
                        help='The baseline model used for obtaining rankings from clustering.')
    parser.add_argument("--normalizations",
                        nargs="+",
                        type=str,
                        help="Normalization methods to choose from: vol_min, vol_sum, vol_max and None.")
    parser.add_argument("--thresholds",
                        nargs="+",
                        type=str,
                        help="Thresholding methods to choose from: sort, std and None.")
    parser.set_defaults(normalizations=['plain'])
    parser.set_defaults(thresholds=['sort'])
    parser.add_argument("--report_normalizations",
                        nargs="+",
                        type=str,
                        help="Normalization methods to generate report.")
    parser.add_argument("--report_thresholds",
                        nargs="+",
                        type=str,
                        help="Thresholding methods to generate report.")
    parser.set_defaults(report_normalizations=['vol_sum','vol_min','vol_max','plain'])
    parser.set_defaults(report_thresholds=['sort', 'std', 'naive'])
    parser.add_argument("--all_methods",
                        nargs="+",
                        type=str,
                        help="Methods to use.")
    parser.set_defaults(all_methods=['btl','DIGRAC'])
    # parser.set_defaults(all_methods=['SpringRank','syncRank','serialRank','btl', 'davidScore',
        # 'eigenvectorCentrality', 'PageRank', 'rankCentrality', 'DIGRAC', 'ib'])
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        help="seeds to generate random graphs.")
    parser.set_defaults(seeds=[10, 20, 30, 40, 50])

    # real data
    parser.add_argument('--season', type=int, default=2009, help='Season for basketball or football.')

    # synthetic model hyperparameters below
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--N', type=int, default=350,
                        help='Number of nodes in the directed stochastic block model.')
    parser.add_argument('--K', type=int, default=5,
                        help='Number of clusters.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training ratio during data split.')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test ratio during data split.')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--tau', type=float, default=0.5,
                        help='The regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    parser.add_argument('--size_ratio', type=float, default=1.5,
                        help='The size ratio of the largest to the smallest block. 1 means uniform sizes. should be at least 1.')
    parser.add_argument('--num_trials', type=int, default=2,
                        help='Number of trials to generate results.')      
    parser.add_argument('--F', default=9,
                        help='Meta-graph adjacency matrix or the number of pairs to consider, array or int.')
    parser.add_argument('--F_style', type=str, default='path',
                        help='Meta-graph adjacency matrix style.')
    parser.add_argument('--ERO_style', type=str, default='uniform',
                        help='ERO rating style, uniform or gamma.')
    parser.add_argument('--ambient', type=int, default=0,
                        help='whether to include ambient nodes in the meta-graph.')
    parser.add_argument('--sp_style', type=str, default='random',
                        help='Spasifying style. Only "random" is supported for now.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--imbalance_coeff', type=float, default=0.0,
                        help='Coefficient of imbalance loss.')
    parser.add_argument('--upset_ratio_coeff', type=float, default=1.0,
                        help='Coefficient of upset ratio loss.')
    parser.add_argument('--upset_margin_coeff', type=float, default=0.0,
                        help='Coefficient of upset margin loss.')
    parser.add_argument('--upset_margin', type=float, default=0.01,
                        help='Margin of upset margin loss.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of iterations to consider for early stopping.')
    parser.add_argument('--fill_val', type=float, default=0.5,
                        help='The value to be filled when we originally have 0, from meta-graph adj to meta-graph to generate data.')
    parser.add_argument('--regenerate_data', action='store_true', help='Whether to force creation of data splits.')
    parser.add_argument('--load_only', action='store_true', help='Whether not to store generated data.')
    parser.add_argument('-AllTrain', '-All', action='store_true', help='Whether to use all data to do gradient descent.')
    parser.add_argument('-SavePred', '-SP', action='store_true', help='Whether to save predicted labels.')
    parser.add_argument('--log_root', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../logs/'), 
                        help='The path saving model.t7 and the training process')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/'), 
                        help='Data set folder.')
    parser.add_argument('--dataset', type=str, default='ERO/', help='Data set selection.')
    
    args = parser.parse_args()
    if args.dataset[-1]!='/':
        args.dataset += '/'

    if args.dataset[:4] != 'DSBM' and args.dataset[:3] != 'ERO':
        args.AllTrain = True
        args.train_ratio = 1
        args.test_ratio = 1
        args.seeds = [10]
    
    if args.dataset[:4] == 'DSBM':
        # calculate the meta-graph adjacency matrix F and the one to generate data: F_data
        args.F = meta_graph_generation(args.F_style, args.K, args.eta, args.ambient, 0)
        args.F_data = meta_graph_generation(args.F_style, args.K, args.eta, args.ambient, args.fill_val)
        default_name_base = args.F_style+ '_' + args.sp_style
        default_name_base += 'p' + str(int(100*args.p)) + 'K' + str(args.K) + 'N' + str(args.N) + 'size_r' + str(int(100*args.size_ratio))
        default_name_base += 'eta' + str(int(100*args.eta)) + 'ambient' + str(args.ambient)
        args.dataset = 'DSBM/' + default_name_base
    elif args.dataset[:3] == 'ERO':
        args.K = 5 # random
        args.F = 3 # random
        default_name_base = 'p' + str(int(100*args.p)) + 'K' + str(args.K) + 'N' + str(args.N)
        default_name_base += 'eta' + str(int(100*args.eta)) + 'style' + str(args.ERO_style)
        args.dataset = 'ERO/' + default_name_base
    elif args.dataset[:10].lower() == 'basketball':
        args.F = 70
        args.K = 20
        args.dataset = 'Basketball_temporal/' + str(args.season)
    elif args.dataset[:16].lower() == 'finer_basketball':
        args.F = 2
        args.K = 20
        args.dataset = 'Basketball_temporal/finer' + str(args.season)
    elif args.dataset[:6].lower() == 'animal':
        args.F = 3
        args.K = 3
        args.dataset = 'Dryad_animal_society/'
    elif args.dataset[:7].lower() == 'finance':
        args.F = 5 # threshold: > 0.7, others have threshold > 0.9
        args.K = 20
    elif args.dataset[:10].lower() == 'headtohead':
        args.F = 39
        args.K = 48
        args.dataset = 'Halo2BetaData/HeadToHead'
    elif args.dataset[:16].lower() == 'faculty_business':
        args.F = 6
        args.K = 5
        args.dataset = 'FacultyHiringNetworks/Business/Business_FM_Full_'
    elif args.dataset[:10].lower() == 'faculty_cs':
        args.F = 8
        args.K = 9
        args.dataset = 'FacultyHiringNetworks/ComputerScience/ComputerScience_FM_Full_'
    elif args.dataset[:15].lower() == 'faculty_history':
        args.F = 22
        args.K = 12
        args.dataset = 'FacultyHiringNetworks/History/History_FM_Full_'
    elif args.dataset[:8].lower() == 'football':
        args.F = 19
        args.K = 9
        args.dataset = 'Football_data_England_Premier_League/England_' + str(args.season) + '_' + str(args.season+1)
    elif args.dataset[:14].lower() == 'finer_football':
        args.F = 4
        args.K = 9
        args.dataset = 'Football_data_England_Premier_League/finerEngland_' + str(args.season) + '_' + str(args.season+1)

    if args.all_methods == ['all_methods_shorter']:
        args.all_methods = ['SpringRank','syncRank','serialRank','btl', 'davidScore',
        'eigenvectorCentrality', 'PageRank', 'rankCentrality', 'SVD_RS', 'SVD_NRS', 'DIGRAC', 'ib']
    elif args.all_methods == ['all_methods_full']:
        args.all_methods = ['SpringRank','syncRank','serialRank','btl', 'davidScore',
        'eigenvectorCentrality', 'PageRank', 'rankCentrality', 'mvr', 'SVD_RS', 'SVD_NRS', 'DIGRAC', 'ib']
    elif args.all_methods == ['all_GNNs']:
        args.all_methods = ['DIGRAC', 'ib']
    elif args.all_methods == ['baselines_shorter']:
        args.all_methods = ['SpringRank','syncRank','serialRank','btl', 'davidScore',
        'eigenvectorCentrality', 'PageRank', 'rankCentrality', 'SVD_RS', 'SVD_NRS']
    elif args.all_methods == ['baselines_full']:
        args.all_methods = ['SpringRank','syncRank','serialRank','btl', 'davidScore',
        'eigenvectorCentrality', 'PageRank', 'rankCentrality', 'mvr', 'SVD_RS', 'SVD_NRS']
    
    if args.train_with in ['anchor_dist', 'anchor_innerproduct']:
        args.optimizer = 'Adam'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    if args.debug:
        args.num_trials = 2
        args.seeds=[10]
        args.epochs = 2
        args.pretrain_epochs = 1
        args.log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../debug_logs/')
    return args
