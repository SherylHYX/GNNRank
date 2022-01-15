from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
torch.autograd.set_detect_anomaly(True)

def set_grad(var, name, change=False):
    def hook(grad):
        if change:
            grad *= 1000
        print(name, grad.max(), grad.min(), grad.norm())
    return hook


class RankingGNNBase(nn.Module):
    r"""The ranking GNN base model.

    Args:
        * **embedding_dim** (int) - Embedding dimension.
        * **Fiedler_layer_num** (int, optional) - The number of single Filder calculation layers, default 3.
        * **alpha** (float, optional) - (Initial) learning rate for the Fiedler step, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
        * **initial_score** (torch.FloatTensor, optional) - Initial guess of scores, default None.
        * **prob_dim** (int, optionial) - Dimension of the probability matrix, default 5.
        * **sigma** (float, optionial) -  (Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.
        * **kwargs (optional): Additional arguments of
            :class:`RankingGNNBase`.
    """

    def __init__(self, embedding_dim: int, Fiedler_layer_num: int=3, alpha: float=0.01, trainable_alpha: bool=False, 
                initial_score: Optional[torch.FloatTensor]=None, prob_dim: int=5, sigma: float=1.0, **kwargs):
        super(RankingGNNBase, self).__init__()
        
        self._anchor_vec = Parameter(torch.FloatTensor(1, embedding_dim).fill_(0))
        self.Q = None
        self.initial_score = initial_score
        self._prob_linear = nn.Linear(embedding_dim, prob_dim)
        self._score_linear = nn.Linear(embedding_dim, 1)
        self._initial_y_baseline = None
        self._log_sigma = Parameter(torch.FloatTensor(1).fill_(sigma))

        self.Fiedler_layers = nn.ModuleList()
        for _ in range(Fiedler_layer_num):
            self.Fiedler_layers.append(Fiedler_Step(alpha, trainable_alpha))
    
    def obtain_score_from_anchor_dist(self) -> torch.FloatTensor:
        self.score = torch.exp(-((self.z - self._anchor_vec).norm(2, 1)/torch.exp(self._log_sigma)) ** 2/self.z.shape[1])[:, None] # Gaussian kernel to serve as similarity, range from 0 to 1
        return self.score

    def obtain_score_from_anchor_innerproduct(self) -> torch.FloatTensor:
        self.score = torch.sigmoid(self._score_linear(self.z)) # to make nonnegative, range from 0 to 1
        return self.score

    def obtain_embedding(self) -> torch.FloatTensor:
        '''
        Obtain embedding matrix.
        '''
        return self.z

    def obtain_similarity_matrix(self) -> torch.FloatTensor:
        '''
        Obtain similarity matrix.
        '''
        sigma = torch.exp(self._log_sigma)
        return torch.exp(-(torch.cdist(self.z, self.z)/sigma) ** 2/self.z.shape[1])


    def obtain_score_from_embedding_similarity(self, start_from: str='dist') -> torch.FloatTensor:
        """
        Obatain similarity matrix from embeddings and use proximal gradient steps. 

        Arg types:
            * **start_from** (str, optional) - Whether to start from a baseline method's, anchor_dist or anchor_innerproduct scores, default dist.
        Return types:
            * **y** (torch.FloatTensor) - Ranking scores of nodes, with shape (num_nodes, ).
        """

        if self.Q is None:
            n = self.z.shape[0]
            device = self.z.device
            u1 = torch.arange(n, 0, -1)
            u2 = torch.arange(n+1, 1, -1)
            u2[0] = 1
            u = torch.sqrt(1 / (u1 * u2)).to(device)
            U = torch.triu(u.view(n, 1).expand(n, n)).to(device)
            s = -torch.sqrt(torch.arange(n-1, 0, -1) / torch.arange(n, 1, -1)).to(device)
            self.Q = U
            self.Q[torch.arange(1, n), torch.arange(0, n-1)] = s

        if self._initial_y_baseline is None:
            self.initial_score = self.initial_score - torch.mean(self.initial_score) 
            # center so that the 0th entry gives zero when multiplied by Q
            self._initial_y_baseline = torch.mm(self.Q[1:], self.initial_score)
            self._initial_y_baseline = F.normalize(self._initial_y_baseline, dim=0)
        
        if start_from == 'baseline':
            y = self._initial_y_baseline.clone()
        else:
            if start_from == 'dist':
                self.score = self.obtain_score_from_anchor_dist()
            elif start_from == 'innerproduct':
                self.score = self.obtain_score_from_anchor_innerproduct()
            else:
                raise ValueError('Please input the correct start_from value from baseline, dist and innerproduct instead of {}!'.format(start_from))
            y = (self.score - torch.mean(self.score))
            # center so that the 0th entry gives zero when multiplied by Q
            y = torch.mm(self.Q[1:], y)
            y = F.normalize(y, dim=0)
            
        # self.z.register_hook(set_grad(self.z, 'gradz'))
        # print('z', self.z.max(), self.z.min())
        S = self.obtain_similarity_matrix()
        # S.register_hook(set_grad(S, 'gradS'))
        # self._sigma.register_hook(set_grad(self._sigma, 'gradsigma'))
        # print('S', S.max(), S.min())
        D = torch.diag(torch.sum(S, 1))
        L = D - S
        
        # d = torch.sum(S, 1) ** -0.5
        # L = torch.eye(d.shape[0]).to(d) - d[:, None] * S * d[None, :]
        '''
        # L.register_hook(set_grad(L))
        _, Q = torch.linalg.eigh(L)
        y = (Q[:, 1:2] + 1)/2
        # y.register_hook(set_grad(y))
    
        return y
        '''
        L = (self.Q @ L @ self.Q.T)[1:][:,1:]
        # print('L', L.max(), L.min())
        # L.register_hook(set_grad(L, 'gradL'))
        for layer in self.Fiedler_layers:
            y = layer(L, y)
        y0 = torch.zeros([1,1], device=y.device)
        y = torch.cat([y0, y], dim=0)
        y = self.Q.T @ y
        y = (y + 1)/2 # to force to be positive, range from 0 to 1
        # print('y', y.max(), y.min())
        # y.register_hook(set_grad(y, 'grady'))
        return y


class Fiedler_Step(torch.nn.Module):
    r"""The Fiedler vector calculation model for a single step.

    Args:
        * **alpha** (float, optional) - (Initial) learning rate, default 1.0.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
    """

    def __init__(self, alpha: float=1.0, trainable_alpha: bool=False):
        super(Fiedler_Step, self).__init__()
        self._trainable_alpha = trainable_alpha
        if self._trainable_alpha:
            self.alpha = Parameter(torch.FloatTensor(1).fill_(alpha))
        else:
            self.alpha = alpha
        self._relu = torch.nn.ReLU()

    def forward(self, L: torch.FloatTensor,
    y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of Fiedler_Vector_Step.

        Arg types:
            * **L** (PyTorch FloatTensor) - The embedding-based Laplacian matrix.
            * **y** (PyTorch FloatTensor) - Initial ranking score vector.

        Return types:
            * **y** (PyTorch FloatTensor) - Updated ranking score vector.
        """
        if self._trainable_alpha:
            self.alpha = Parameter(self._relu(self.alpha))
        Ly = torch.matmul((L + torch.transpose(L, 0, 1)), y)
        # Ly.register_hook(set_grad(Ly, 'gradLY'))
        # print('y', y.norm(), y.max(), y.min())
        y_new = y - self.alpha * Ly/y.shape[0] # divide by number of nodes to avoid driven by this term
        # y_new.register_hook(set_grad(y_new, 'gradYNew_in1'))
        # print(y_new.max(), y_new.min(), y.max(), y.min(), Ly.max(), Ly.min())
        y_new = F.normalize(y_new, dim=0)
        # y_new.register_hook(set_grad(y_new, 'gradYNew_in2'))
        #print('alpha', self.alpha)
        
        y = y_new
        # print('Fiedler check',y.T @ L @ y, torch.linalg.svdvals(L)[-2])
        return y
        '''
        if self._trainable_alpha:
            self.alpha = Parameter(self._relu(self.alpha))
        y_new = y - self.alpha * torch.matmul((L + torch.transpose(L, 0, 1)), y)/y.shape[0] # divide by number of nodes to avoid driven by this term
        y_new = F.normalize(y_new, dim=0)
        y = y_new
        print(y.T @ L @ y, torch.linalg.svdvals(L)[-2])
        return y
        
        for alpha in [self.alpha, 0.5*self.alpha, 0.2*self.alpha, 0.1*self.alpha]:
            y_new = y - alpha * torch.matmul((L + torch.transpose(L, 0, 1)), y)
            y_new = F.normalize(y_new, dim=0)
            # print((torch.matmul(torch.transpose(y, 0, 1), torch.matmul(L, y)))[0,0].item(), torch.matmul(torch.transpose(y_new, 0, 1), torch.matmul(L, y_new))[0,0].item())
            if (y.T @ L @ y - y_new.T @ L @y_new)[0,0].item() > 0:
                y = y_new
                return y
        return y # then nothing changes
        '''