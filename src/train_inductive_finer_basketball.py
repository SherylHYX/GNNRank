# external files
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from texttable import Texttable
from scipy.stats import kendalltau, rankdata

# internal files
from utils import write_log, scipy_sparse_to_torch_sparse, get_powers_sparse
from metrics import print_performance_mean_std, calculate_upsets
from GNN_models import DIGRAC_Ranking, DiGCN_Inception_Block_Ranking
from param_parser import parameter_parser
from preprocess import load_data
from get_adj import get_second_directed_adj
from SpringRank import SpringRank
from comparison import syncRank_angle, syncRank, serialRank, btl, davidScore, eigenvectorCentrality, PageRank, rankCentrality, mvr
from comparison import SVD_RS, SVD_NRS, serialRank_matrix

GNN_variant_names = ['dist', 'innerproduct', 'proximal_dist', 'proximal_innerproduct', 'proximal_baseline']
NUM_GNN_VARIANTS = len(GNN_variant_names) # number of GNN variants for each architecture

upset_choices = ['upset_simple', 'upset_ratio', 'upset_naive']
NUM_UPSET_CHOICES = len(upset_choices)
args = parameter_parser()
args.num_trials = 10
args.upset_margin_coeff = 1
args.trainable_alpha = True
args.baseline = 'syncRank'
args.lr = 0.01
args.hidden = 8
args.F = 2
args.K = 20
args.dataset = 'Basketball_temporal/finer' + str(args.season)
args.epochs = 0
args.train_ratio = 1
args.test_ratio = 1
args.seeds = [10]
args.train_with = 'proximal_baseline'
args.pretrain_with = 'serial_similarity'
args.all_methods = ['ib']
args.AllTrain = True
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
compare_names_all = []
for method_name in args.all_methods:
    if method_name not in ['DIGRAC', 'ib']:
        compare_names_all.append(method_name)
    else:
        for GNN_type in GNN_variant_names:
            compare_names_all.append(method_name+'_'+GNN_type)

def evalutaion(logstr, score, A_torch, label_np, val_index, test_index, SavePred, save_path, split, identifier_str):
    kendalltau_full = np.zeros((3, 2))
    kendalltau_full[:] = np.nan
    if score.min().detach().item() < 0:
        if score.min().detach().item() > -1:
            score = (score + 1)/2
        else:
            score = torch.sigmoid(score)
    upset1 = calculate_upsets(A_torch, score)
    upset2 = calculate_upsets(torch.transpose(A_torch, 0, 1), score)
    if upset1.detach().item() < upset2.detach().item():
        upset_ratio = upset1
    else:
        upset_ratio = upset2
        score = -score
    pred_label = rankdata(-score.detach().cpu().numpy(), 'min')
    upset_simple = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(args.device), style='simple').detach().item()
    upset_naive = calculate_upsets(A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(args.device), style='naive').detach().item()
    upset_full = [upset_simple, upset_ratio.detach().item(), upset_naive]
    if SavePred:
        np.save(save_path+identifier_str+'_pred'+str(split), pred_label)
        np.save(save_path+identifier_str+'_scores'+str(split), score.detach().cpu().numpy())

    logstr += '\n From ' + identifier_str + ':,'
    if label_np is not None:
        # test
        tau, p_value = kendalltau(pred_label[test_index], label_np[test_index])
        outstrtest = 'Test kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[0] = [tau, p_value]
        
        # val
        tau, p_value = kendalltau(pred_label[val_index], label_np[val_index])
        outstrval = 'Validation kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[1] = [tau, p_value]
        
        
        # all
        tau, p_value = kendalltau(pred_label, label_np)
        outstrall = 'All kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[2] = [tau, p_value]

        logstr += outstrtest + outstrval + outstrall
    logstr += 'upset simple:,{:.6f},upset ratio:,{:.6f},upset naive:,{:.6f},'.format(upset_simple, upset_ratio.detach().item(), upset_naive)
    return logstr, upset_full, kendalltau_full


class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args, random_seed, save_name_base):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device
        self.random_seed = random_seed

        self.label, self.train_mask, self.val_mask, self.test_mask, self.features, self.A = load_data(args, random_seed)
        self.features = torch.FloatTensor(self.features).to(args.device)
        self.args.N = self.A.shape[0]
        self.A_torch = torch.FloatTensor(self.A.toarray()).to(device)
        
        self.nfeat = self.features.shape[1]
        if self.label is not None:
            self.label = torch.LongTensor(self.label).to(args.device)
            self.label_np = self.label.to('cpu')
            if  self.args.dataset[:3] != 'ERO':
                self.args.K = int(self.label_np.max() - self.label_np.min() + 1)
        else:
            self.label_np = None
        self.num_clusters = self.args.K

        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        

        save_name = save_name_base + 'Seed' + str(random_seed)

        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, date_time)
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.log_root, 'Basketball_temporal/finer1985', save_name)
        all_subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        self.log_dir = os.path.join(log_dir, latest_subdir)
        if self.args.epochs == 0:
            self.log_path = self.log_dir

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.splits = self.args.num_trials
        if self.test_mask is not None and self.test_mask.ndim == 1:
            self.train_mask = np.repeat(
                self.train_mask[:, np.newaxis], self.splits, 1)
            self.val_mask = np.repeat(
                self.val_mask[:, np.newaxis], self.splits, 1)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)
        write_log(vars(args), self.log_path)  # write the setting


    def train(self, model_name):
        #################################
        # training and evaluation
        #################################
        if model_name not in ['DIGRAC', 'ib']:
            kendalltau_full, upset_full = self.non_nn(model_name)
            kendalltau_full_latest = kendalltau_full.copy()
            upset_full_latest = upset_full.copy()
        else:
            if self.args.upset_ratio_coeff + self.args.upset_margin_coeff == 0:
                raise ValueError('Incorrect loss combination!')
            # (the last two dimensions) rows: test, val, all; cols: kendall tau, kendall p value
            kendalltau_full = np.zeros([NUM_GNN_VARIANTS, self.splits, 3, 2])
            kendalltau_full[:] = np.nan
            kendalltau_full_latest = kendalltau_full.copy()

            upset_full = np.zeros([NUM_GNN_VARIANTS, self.splits, NUM_UPSET_CHOICES])
            upset_full[:] = np.nan
            upset_full_latest = upset_full.copy()
            
            args = self.args
            A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
            if model_name == 'DIGRAC':
                norm_A = get_powers_sparse(self.A, hop=1, tau=self.args.tau)[
                    1].to(self.args.device)
                norm_At = get_powers_sparse(self.A.transpose(), hop=1, tau=self.args.tau)[
                    1].to(self.args.device)
            elif model_name == 'ib':
                edge_index = torch.LongTensor(self.A.nonzero())
                edge_weights = torch.FloatTensor(self.A.data)
                edge_index1 = edge_index.clone().to(self.args.device)
                edge_weights1 = edge_weights.clone().to(self.args.device)
                edge_index2, edge_weights2 = get_second_directed_adj(edge_index, self.features.shape[0],self.features.dtype,
                edge_weights)
                edge_index2 = edge_index2.to(self.args.device)
                edge_weights2 = edge_weights2.to(self.args.device)
                edge_index = (edge_index1, edge_index2)
                edge_weights = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
            for split in range(self.splits):
                if self.args.baseline == 'SpringRank':
                    score = SpringRank(self.A,alpha=0,l0=1,l1=1)
                elif self.args.baseline == 'serialRank':
                    score = serialRank(self.A)
                elif self.args.baseline == 'btl':
                    score = btl(self.A)
                elif self.args.baseline == 'davidScore':
                    score = davidScore(self.A)
                elif self.args.baseline == 'eigenvectorCentrality':
                    score = eigenvectorCentrality(self.A)
                elif self.args.baseline == 'PageRank':
                    score = PageRank(self.A)
                elif self.args.baseline == 'rankCentrality':
                    score = rankCentrality(self.A)
                elif self.args.baseline == 'syncRank':
                    score = syncRank_angle(self.A) # scores
                elif self.args.baseline == 'SVD_RS':
                    score = SVD_RS(self.A)
                elif self.args.baseline == 'SVD_NRS':
                    score = SVD_NRS(self.A)
                else:
                    raise NameError('Please input the correct baseline model name from:\
                        SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                        PageRank, rankCentrality, SVD_RS, SVD_NRS instead of {}!'.format(self.args.baseline))
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)
                if score.min() < 0:
                    score_torch = torch.sigmoid(score_torch)
                upset1 = calculate_upsets(self.A_torch, score_torch)
                upset2 = calculate_upsets(torch.transpose(self.A_torch, 0, 1), score_torch)
                
                if upset1.detach().item() > upset2.detach().item():
                    score = -score
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)

                if model_name == 'DIGRAC':
                    model = DIGRAC_Ranking(num_features=self.nfeat, dropout=self.args.dropout, hop=self.args.hop, fill_value=self.args.tau, 
                embedding_dim=self.args.hidden*2, Fiedler_layer_num=self.args.Fiedler_layer_num, alpha=self.args.alpha, 
                trainable_alpha=self.args.trainable_alpha, initial_score=score_torch, prob_dim=self.num_clusters, sigma=self.args.sigma).to(self.args.device)
                    model_path = os.path.join(self.log_dir, 'DIGRAC_model'+str(split)+'.t7')
                    model.load_state_dict(torch.load(model_path))
                elif model_name == 'ib':
                    model = DiGCN_Inception_Block_Ranking(num_features=self.nfeat, dropout=self.args.dropout,
                embedding_dim=self.args.hidden*2, Fiedler_layer_num=self.args.Fiedler_layer_num, alpha=self.args.alpha, 
                trainable_alpha=self.args.trainable_alpha, initial_score=score_torch, prob_dim=self.num_clusters, sigma=self.args.sigma).to(self.args.device)
                else:
                    raise NameError('Please input the correct model name from:\
                        SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                        PageRank, rankCentrality, mvr, DIGRAC, ib, instead of {}!'.format(model_name))

                if self.args.optimizer == 'Adam':
                    opt = optim.Adam(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                elif self.args.optimizer == 'SGD':
                    opt = optim.SGD(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                else:
                    raise NameError('Please input the correct optimizer name, Adam or SGD!')
                M = self.A_torch

                if self.test_mask is not None:
                    train_index = self.train_mask[:, split]
                    val_index = self.val_mask[:, split]
                    test_index = self.test_mask[:, split]
                    if args.AllTrain:
                        # to use all nodes
                        train_index[:] = True
                        val_index[:] = True
                        test_index[:] = True
                    train_A = scipy_sparse_to_torch_sparse(self.A[train_index][:, train_index]).to(self.args.device)
                    val_A = scipy_sparse_to_torch_sparse(self.A[val_index][:, val_index]).to(self.args.device)
                    test_A = scipy_sparse_to_torch_sparse(self.A[test_index][:, test_index]).to(self.args.device)
                else:
                    train_index = np.ones(args.N, dtype=bool)
                    val_index = train_index
                    test_index = train_index
                    train_A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
                    val_A = train_A
                    test_A = train_A
                #################################
                # Train/Validation/Test
                #################################
                best_val_loss = 1000.0
                early_stopping = 0
                log_str_full = ''
                train_with = self.args.train_with
                if self.args.pretrain_with == 'serial_similarity' and args.train_with[:8] == 'proximal':
                    serialRank_mat = serialRank_matrix(self.A[train_index][:, train_index])
                    serialRank_mat = serialRank_mat/max(0.1, serialRank_mat.max())
                    serial_matrix_train = torch.FloatTensor(serialRank_mat.toarray()).to(self.args.device)

                for epoch in range(args.epochs):
                    if self.args.optimizer == 'Adam' and epoch == self.args.pretrain_epochs and self.args.train_with[:8] == 'proximal':
                        opt = optim.SGD(model.parameters(), lr=10*self.args.lr,
                                        weight_decay=self.args.weight_decay)
                    start_time = time.time()
                    ####################
                    # Train
                    ####################

                    model.train()
                    if model_name == 'DIGRAC':
                        prob = model(norm_A, norm_At, self.features)
                    elif model_name == 'ib':
                        prob = model(edge_index, edge_weights, self.features)
                    if train_with == 'dist' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                        score = model.obtain_score_from_dist()
                    elif train_with == 'innerproduct' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                        score = model.obtain_score_from_innerproduct()
                    else:
                        score = model.obtain_score_from_embedding_similarity(train_with[9:])
                       
                    if self.args.upset_ratio_coeff > 0:
                        train_loss_upset_ratio = calculate_upsets(M[train_index][:,train_index], score[train_index])               
                    else:
                        train_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    if self.args.upset_margin_coeff > 0:
                        train_loss_upset_margin = calculate_upsets(M[train_index][:,train_index], score[train_index], style='margin', margin=self.args.upset_margin)               
                    else:
                        train_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    

                    train_loss = self.args.upset_ratio_coeff * train_loss_upset_ratio + self.args.upset_margin_coeff * train_loss_upset_margin
                    if self.args.pretrain_with == 'serial_similarity' and epoch < self.args.pretrain_epochs and args.train_with[:8] == 'proximal':
                        pretrain_outside_loss = torch.mean((model.obtain_similarity_matrix()[train_index][:, train_index] - serial_matrix_train) ** 2)
                        train_loss += pretrain_outside_loss
                        outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f}, pretrian outside loss: {:6f},'.format(train_loss.detach().item(),
                        train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item(), 
                        pretrain_outside_loss.detach().item())
                    else:
                        outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(train_loss.detach().item(),
                        train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item())
                    opt.zero_grad()
                    try:
                        train_loss.backward()
                    except RuntimeError:
                        log_str = '{} trial {} RuntimeError!'.format(model_name, split)
                        log_str_full += log_str + '\n'
                        print(log_str)
                        if not os.path.isfile(self.log_path + '/'+model_name+'_model'+str(split)+'.t7'):
                                torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model'+str(split)+'.t7')
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model_latest'+str(split)+'.t7')
                        break
                    opt.step()
                    ####################
                    # Validation
                    ####################
                    model.eval()

                    if model_name == 'DIGRAC':
                        prob = model(norm_A, norm_At, self.features)
                    elif model_name == 'ib':
                        prob = model(edge_index, edge_weights, self.features)
                    if train_with == 'dist' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                        score = model.obtain_score_from_dist()
                    elif train_with == 'innerproduct' or (epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                        score = model.obtain_score_from_innerproduct()
                    else:
                        score = model.obtain_score_from_embedding_similarity(train_with[9:])

                    if self.args.upset_ratio_coeff > 0:
                        val_loss_upset_ratio = calculate_upsets(M[val_index][:,val_index], score[val_index])               
                    else:
                        val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    if self.args.upset_margin_coeff > 0:
                        val_loss_upset_margin = calculate_upsets(M[val_index][:,val_index], score[val_index], style='margin', margin=self.args.upset_margin)               
                    else:
                        val_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    
                    val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin


                    outstrval = 'val loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(val_loss.detach().item(),
                    val_loss_upset_ratio.detach().item(), val_loss_upset_margin.detach().item())

                    duration = "---, {:.4f}, seconds ---".format(
                        time.time() - start_time)
                    log_str = ("{}, / {} epoch,".format(epoch, args.epochs)) + \
                        outstrtrain+outstrval+duration
                    log_str_full += log_str + '\n'
                    print(log_str)
                    
                    ####################
                    # Save weights
                    ####################
                    save_perform = val_loss.detach().item()
                    if save_perform <= best_val_loss:
                        early_stopping = 0
                        best_val_loss = save_perform
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model'+str(split)+'.t7')
                    else:
                        early_stopping += 1
                    if early_stopping > args.early_stopping or epoch == (args.epochs-1):
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model_latest'+str(split)+'.t7')
                        break

                status = 'w'
                if os.path.isfile(self.log_path + '/'+model_name+'_log'+str(split)+'.csv'):
                    status = 'a'
                with open(self.log_path + '/'+model_name+'_log'+str(split)+'.csv', status) as file:
                    file.write(log_str_full)
                    file.write('\n')
                    status = 'a'

                ####################
                # Testing
                ####################
                base_save_path = self.log_path + '/'+model_name
                logstr = ''
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model'+str(split)+'.t7'))
                model.eval()

                if model_name == 'DIGRAC':
                    prob = model(norm_A, norm_At, self.features)
                elif model_name == 'ib':
                    prob = model(edge_index, edge_weights, self.features)
                if train_with == 'dist':
                    score_model = model.obtain_score_from_dist()
                elif train_with == 'innerproduct':
                    score_model = model.obtain_score_from_innerproduct()
                else:
                    score_model = model.obtain_score_from_embedding_similarity(train_with[9:])

                if self.args.upset_ratio_coeff > 0:
                    val_loss_upset_ratio = calculate_upsets(M[val_index][:,val_index], score_model[val_index]) 
                    test_loss_upset_ratio = calculate_upsets(M[test_index][:,test_index], score_model[test_index])  
                    all_loss_upset_ratio = calculate_upsets(M, score_model)                
                else:
                    val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                if self.args.upset_margin_coeff > 0:
                    val_loss_upset_margin = calculate_upsets(M[val_index][:,val_index], score_model[val_index], style='margin', margin=self.args.upset_margin) 
                    test_loss_upset_margin = calculate_upsets(M[test_index][:,test_index], score_model[test_index], style='margin', margin=self.args.upset_margin)               
                    all_loss_upset_margin = calculate_upsets(M, score_model, style='margin', margin=self.args.upset_margin)                             
                else:
                    val_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin
                test_loss = self.args.upset_ratio_coeff * test_loss_upset_ratio + self.args.upset_margin_coeff * test_loss_upset_margin
                all_loss = self.args.upset_ratio_coeff * all_loss_upset_ratio + self.args.upset_margin_coeff * all_loss_upset_margin

                logstr += 'Final results for {}:,'.format(model_name)
                logstr += 'Best val loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())

                score = model.obtain_score_from_dist()
                logstr, upset_full[0, split], kendalltau_full[0, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, 'dist')
                score = model.obtain_score_from_innerproduct()
                logstr, upset_full[1, split], kendalltau_full[1, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, 'innerproduct')
                for ind, start_from in enumerate(['dist', 'innerproduct', 'baseline']):
                    score = model.obtain_score_from_proximal(start_from)
                    logstr, upset_full[2 + ind, split], kendalltau_full[2 + ind, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                        base_save_path, split, 'proximal_'+start_from)
                
                

                # latest
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model_latest'+str(split)+'.t7'))
                model.eval()

                if model_name == 'DIGRAC':
                    prob = model(norm_A, norm_At, self.features)
                elif model_name == 'ib':
                    prob = model(edge_index, edge_weights, self.features)
                if train_with == 'dist':
                    score_model = model.obtain_score_from_dist()
                elif train_with == 'innerproduct':
                    score_model = model.obtain_score_from_innerproduct()
                else:
                    score_model = model.obtain_score_from_embedding_similarity(train_with[9:])
                
                if self.args.upset_ratio_coeff > 0:
                    val_loss_upset_ratio = calculate_upsets(M[val_index][:,val_index], score_model[val_index]) 
                    test_loss_upset_ratio = calculate_upsets(M[test_index][:,test_index], score_model[test_index])  
                    all_loss_upset_ratio = calculate_upsets(M, score_model)                
                else:
                    val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                if self.args.upset_margin_coeff > 0:
                    val_loss_upset_margin = calculate_upsets(M[val_index][:,val_index], score_model[val_index], style='margin', margin=self.args.upset_margin) 
                    test_loss_upset_margin = calculate_upsets(M[test_index][:,test_index], score_model[test_index], style='margin', margin=self.args.upset_margin)               
                    all_loss_upset_margin = calculate_upsets(M, score_model, style='margin', margin=self.args.upset_margin)                             
                else:
                    val_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin
                test_loss = self.args.upset_ratio_coeff * test_loss_upset_ratio + self.args.upset_margin_coeff * test_loss_upset_margin
                all_loss = self.args.upset_ratio_coeff * all_loss_upset_ratio + self.args.upset_margin_coeff * all_loss_upset_margin

                logstr += 'Latest val loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())


                
                score = model.obtain_score_from_dist()
                logstr, upset_full_latest[0, split], kendalltau_full_latest[0, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, 'dist_latest')
                score = model.obtain_score_from_innerproduct()
                logstr, upset_full_latest[1, split], kendalltau_full_latest[1, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, 'innerproduct_latest')
                for ind, start_from in enumerate(['dist', 'innerproduct', 'baseline']):
                    score = model.obtain_score_from_proximal(start_from)
                    logstr, upset_full_latest[2 + ind, split], kendalltau_full_latest[2 + ind, split] = evalutaion(logstr, score, self.A_torch, self.label_np, val_index, test_index, self.args.SavePred, \
                        base_save_path, split, 'proximal_'+start_from+'_latest')

                print(logstr)

                with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', status) as file:
                    file.write(logstr)
                    file.write('\n')

                torch.cuda.empty_cache()
        return kendalltau_full, kendalltau_full_latest, upset_full, upset_full_latest

    def non_nn(self, model_name):
        #################################
        # training and evaluation for non-NN methods
        #################################
        # rows: test, val, all; cols: kendall tau, kendall p value
        kendalltau_full = np.zeros([self.splits, 3, 2])
        kendalltau_full[:] = np.nan
        upset_full = np.zeros([self.splits, NUM_UPSET_CHOICES])
        upset_full[:] = np.nan
        A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
        
        for split in range(self.splits):
            if self.test_mask is not None:
                val_index = self.val_mask[:, split]
                test_index = self.test_mask[:, split]
                if args.AllTrain:
                    # to use all nodes
                    val_index[:] = True
                    test_index[:] = True
            else:
                val_index = np.ones(args.N, dtype=bool)
                test_index = val_index

            ####################
            # Testing
            ####################
            logstr = ''

            if model_name == 'SpringRank':
                score = SpringRank(self.A,alpha=0,l0=1,l1=1)
            elif model_name == 'serialRank':
                score = serialRank(self.A)
            elif model_name == 'btl':
                score = btl(self.A)
            elif model_name == 'davidScore':
                score = davidScore(self.A)
            elif model_name == 'eigenvectorCentrality':
                score = eigenvectorCentrality(self.A)
            elif model_name == 'PageRank':
                score = PageRank(self.A)
            elif model_name == 'rankCentrality':
                score = rankCentrality(self.A)
            elif model_name == 'syncRank':
                pred_label = syncRank(self.A)
                score = syncRank_angle(self.A) # scores
            elif model_name == 'mvr':
                pred_label = mvr(self.A)
            elif model_name == 'SVD_RS':
                score = SVD_RS(self.A)
            elif model_name == 'SVD_NRS':
                score = SVD_NRS(self.A)
            else:
                raise NameError('Please input the correct model name from:\
                    SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                    PageRank, rankCentrality, mvr, SVD_RS, SVD_NRS, DIGRAC, ib, instead of {}!'.format(model_name))
            if model_name not in ['mvr']:
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)
                if score.min() < 0:
                    if score.min()  > -1:
                        score_torch = (score_torch + 1)/2
                    else:
                        score_torch = torch.sigmoid(score_torch)
                upset1 = calculate_upsets(self.A_torch, score_torch)
                upset2 = calculate_upsets(torch.transpose(self.A_torch, 0, 1), score_torch)
                
                if model_name not in ['syncRank']:
                    if upset1.detach().item() > upset2.detach().item():
                        upset_ratio = upset2.detach().item()
                        score = -score
                    else:
                        upset_ratio = upset1.detach().item()
                    pred_label = rankdata(-score, 'min')
                else:
                    if upset1.detach().item() < upset2.detach().item():
                        upset_ratio = upset1.detach().item()
                        pred_label = 1 + pred_label.max()-pred_label
                    else:
                        upset_ratio = upset2.detach().item()
            else:
                upset_ratio = np.nan
                score = np.nan

            upset_simple = calculate_upsets(self.A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.args.device), style='simple').detach().item()
            upset_naive = calculate_upsets(self.A_torch, torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(self.args.device), style='naive').detach().item()
            upset_full[split] = [upset_simple, upset_ratio, upset_naive]
            logstr += 'upset simple:,{:.6f},upset ratio:,{:.6f},upset naive:,{:.6f},'.format(upset_simple, upset_ratio, upset_naive)

            if self.args.SavePred:
                np.save(self.log_path + '/'+model_name+
                        '_pred'+str(split), pred_label)
                np.save(self.log_path + '/'+model_name+
                        '_scores'+str(split), score)
            
            print('Final results for {}:'.format(model_name))
            if self.label is not None:
                # test
                tau, p_value = kendalltau(pred_label[test_index], self.label_np[test_index])
                outstrtest = 'Test kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
                kendalltau_full[split, 0] = [tau, p_value]
                
                # val
                tau, p_value = kendalltau(pred_label[val_index], self.label_np[val_index])
                outstrval = 'Validation kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
                kendalltau_full[split, 1] = [tau, p_value]
                
                
                # all
                tau, p_value = kendalltau(pred_label, self.label_np)
                outstrall = 'All kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
                kendalltau_full[split, 2] = [tau, p_value]
                    
                logstr += outstrtest + outstrval + outstrall

            print(logstr)

            with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', 'a') as file:
                file.write(logstr)
                file.write('\n')
        return kendalltau_full, upset_full


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays_inductive/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays_inductive/'+args.dataset)


kendalltau_res = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 2])
kendalltau_res_latest = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 2])

final_upset = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), NUM_UPSET_CHOICES])
final_upset[:] = np.nan
final_upset_latest = final_upset.copy()

method_str = ''
for method_name in args.all_methods:
    method_str += method_name

default_name_base = ''
if 'DIGRAC' in args.all_methods or 'ib' in args.all_methods:
    default_name_base += 'K' + str(args.K) + 'dropout' + str(int(100*args.dropout))
    default_name_base += 'ratio_coe' + str(int(100*args.upset_ratio_coeff)) + 'margin_coe' + str(int(100*args.upset_margin_coeff)) 
    if args.upset_margin_coeff > 0:
        default_name_base += 'margin' + str(int(100*args.upset_margin)) 
    default_name_base += 'with' + str(args.train_with)  + 'Fiedler' + str(args.Fiedler_layer_num) + 'sigma' + str(int(100*args.sigma))
    default_name_base += 'alpha' + str(int(100*args.alpha))
    if args.train_with[:8] == 'proximal':
        default_name_base += 'train_alpha' + str(args.trainable_alpha)
    default_name_base += 'hid' + str(args.hidden) + 'lr' + str(int(1000*args.lr))
    default_name_base += 'use' + str(args.baseline)
    if args.pretrain_epochs > 0 and args.train_with[:8] == 'proximal':
        default_name_base +=  'pre' + str(args.pretrain_with) + str(int(args.pretrain_epochs))
save_name_base = default_name_base

default_name_base +=  'trials' + str(args.num_trials) + 'train_r' + str(int(100*args.train_ratio)) + 'test_r' + str(int(100*args.test_ratio)) + 'All' + str(args.AllTrain)
save_name_base = default_name_base
if args.dataset[:3] == 'ERO':
    default_name_base += 'seeds' + '_'.join([str(value) for value in np.array(args.seeds).flatten()])
save_name = default_name_base


current_seed_ind = 0
for random_seed in args.seeds:
    current_ind = 0
    trainer = Trainer(args, random_seed, save_name_base)
    for method_name in args.all_methods:
        kendalltau_full, kendalltau_full_latest, upset_full, upset_full_latest = trainer.train(method_name)
        if method_name not in ['DIGRAC', 'ib']:
            kendalltau_res[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full
            kendalltau_res_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full_latest
            final_upset[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += 1
        else:
            kendalltau_res[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full
            kendalltau_res_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = kendalltau_full_latest
            final_upset[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += NUM_GNN_VARIANTS
    current_seed_ind += args.num_trials

# print results and save results to arrays
t = Texttable(max_width=120)
t.add_rows([["Parameter", "K", "trainable_alpha", "Fiedler_layer_num", "num_trials", "alpha", "dropout", \
"upset_ratio_coeff", "upset_margin_coeff", "margin","train_with", "baseline", "pretrain with", "pretrain epochs"],
["Values",args.K, args.trainable_alpha, args.Fiedler_layer_num, args.num_trials, args.alpha, args.dropout,
args.upset_ratio_coeff, args.upset_margin_coeff, args.upset_margin, args.train_with, 
args.baseline, args.pretrain_with, args.pretrain_epochs]])
print(t.draw())

for save_dir_name in ['kendalltau', 'upset']:
    if os.path.isdir(os.path.join(dir_name,save_dir_name,method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name,method_str))
        except FileExistsError:
            print('Folder exists for best {}!'.format(save_dir_name))
    if os.path.isdir(os.path.join(dir_name,save_dir_name+'_latest',method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name+'_latest',method_str))
        except FileExistsError:
            print('Folder exists for latest {}!'.format(save_dir_name))

np.save(os.path.join(dir_name,'kendalltau',method_str,save_name), kendalltau_res)
np.save(os.path.join(dir_name,'kendalltau_latest',method_str,save_name), kendalltau_res_latest)
np.save(os.path.join(dir_name,'upset',method_str,save_name), final_upset)
np.save(os.path.join(dir_name,'upset_latest',method_str,save_name), final_upset_latest)


new_shape_upset = (args.num_trials*len(args.seeds), len(compare_names_all), NUM_UPSET_CHOICES)
metric_names = ['test kendall tau', 'test kendall p', 'val kendall tau', 'val kendall p', \
'all kendall tau', 'all kendall p'] + upset_choices
new_shape_kendalltau = (args.num_trials*len(args.seeds), len(compare_names_all), 6)

print_performance_mean_std(args.dataset+'_latest', np.concatenate((kendalltau_res_latest.swapaxes(0,1).reshape(new_shape_kendalltau),
                final_upset_latest.swapaxes(0,1).reshape(new_shape_upset)), axis=-1), 
                compare_names_all, metric_names, False)

print('\n')

print_performance_mean_std(args.dataset+'_best', np.concatenate((kendalltau_res.swapaxes(0,1).reshape(new_shape_kendalltau),
                final_upset.swapaxes(0,1).reshape(new_shape_upset)), axis=-1), 
                compare_names_all, metric_names, False)
