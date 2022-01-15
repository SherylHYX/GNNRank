from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from RankingGNNBase import RankingGNNBase
from DiGCN_Inception_Block import DiGCN_InceptionBlock as InceptionBlock

import torch
from torch.nn.parameter import Parameter

class DIMPA(torch.nn.Module):
    r"""The directed mixed-path aggregation model.

    Args:
        hop (int): Number of hops to consider.
    """

    def __init__(self, hop: int):
        super(DIMPA, self).__init__()
        self._hop = hop
        self._w_s = Parameter(torch.FloatTensor(hop + 1, 1))
        self._w_t = Parameter(torch.FloatTensor(hop + 1, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        self._w_s.data.fill_(1.0)
        self._w_t.data.fill_(1.0)

    def forward(self, x_s: torch.FloatTensor, x_t: torch.FloatTensor,
                A: Union[torch.FloatTensor, torch.sparse_coo_tensor], 
                At: Union[torch.FloatTensor, torch.sparse_coo_tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of DIMPA.

        Arg types:
            * **x_s** (PyTorch FloatTensor) - Souce hidden representations.
            * **x_t** (PyTorch FloatTensor) - Target hidden representations.
            * **A** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized adjacency matrix.
            * **At** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Tranpose of column-normalized adjacency matrix.

        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim).
        """
        feat_s = self._w_s[0]*x_s
        feat_t = self._w_t[0]*x_t
        curr_s = x_s.clone()
        curr_t = x_t.clone()
        for h in range(1, 1+self._hop):
            curr_s = torch.matmul(A, curr_s)
            curr_t = torch.matmul(At, curr_t)
            feat_s += self._w_s[h]*curr_s
            feat_t += self._w_t[h]*curr_t

        feat = torch.cat([feat_s, feat_t], dim=1)  # concatenate results

        return feat

class DIGRAC_Ranking(RankingGNNBase):
    r"""The directed graph clustering model from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **hop** (int): Number of hops to consider.
        * **fill_value** (float): Value for added self-loops.
        * **embedding_dim** (int) - Embedding dimension.
        * **Fiedler_layer_num** (int, optional) - The number of single Filder calculation layers, default 3.
        * **alpha** (float, optional) - (Initial) learning rate for the Fiedler step, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
        * **initial_score** (torch.FloatTensor, optional) - Initial guess of scores, default None.
        * **prob_dim** (int, optionial) - Dimension of the probability matrix, default 5.
        * **sigma** (float, optionial) - (Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.
        * **kwargs (optional): Additional arguments of
            :class:`RankingGNNBase`.
    """

    def __init__(self, num_features: int, dropout: float, hop: int, fill_value: float, 
                embedding_dim: int, Fiedler_layer_num: int=3, alpha: float=0.01, 
                trainable_alpha: bool=False, initial_score: Optional[torch.FloatTensor]=None, prob_dim: int=5, sigma: float=1.0, **kwargs):
        super(DIGRAC_Ranking, self).__init__(embedding_dim, Fiedler_layer_num, alpha, 
                trainable_alpha, initial_score, prob_dim, sigma, **kwargs)
        hidden = int(embedding_dim/2)
        nh1 = hidden
        nh2 = hidden
        self._num_clusters = int(prob_dim)
        self._w_s0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_s1 = Parameter(torch.FloatTensor(nh1, nh2))
        self._w_t0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_t1 = Parameter(torch.FloatTensor(nh1, nh2))

        self._dimpa = DIMPA(hop)
        self._relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_s0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_s1, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t1, gain=1.414)

    def forward(self, A: Union[torch.FloatTensor, torch.sparse_coo_tensor], At: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the DIGRAC from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.
        Arg types:
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
        Return types:
            * **output** (PyTorch FloatTensor) - Class probabilities for all nodes, 
                with shape (num_nodes, prob_dim).
        """
        # MLP
        x_s = torch.mm(features, self._w_s0)
        x_s = self._relu(x_s)
        x_s = self.dropout(x_s)
        x_s = torch.mm(x_s, self._w_s1)

        x_t = torch.mm(features, self._w_t0)
        x_t = self._relu(x_t)
        x_t = self.dropout(x_t)
        x_t = torch.mm(x_t, self._w_t1)

        self.z = self._dimpa(x_s, x_t, A, At)

        output = self._prob_linear(self.z)
        output = F.softmax(output, dim=1)
        return output


class DiGCN_Inception_Block_Ranking(RankingGNNBase):
    r"""An implementation of the DiGCN model with inception blocks for ranking from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **embedding_dim** (int) - Embedding dimension.
        * **Fiedler_layer_num** (int, optional) - The number of single Filder calculation layers, default 3.
        * **alpha** (float, optional) - (Initial) learning rate for the Fiedler step, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
        * **initial_score** (torch.FloatTensor, optional) - Initial guess of scores, default None.
        * **prob_dim** (int, optionial) - Dimension of the probability matrix, default 5.
        * **sigma** (float, optionial) - (Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.
        * **kwargs (optional): Additional arguments of
            :class:`RankingGNNBase`.
    """
    def __init__(self, num_features: int, dropout: float, 
                embedding_dim: int, Fiedler_layer_num: int=3, alpha: float=0.01, 
                trainable_alpha: bool=False, initial_score: Optional[torch.FloatTensor]=None, prob_dim: int=5, sigma: float=1.0, **kwargs):
        super(DiGCN_Inception_Block_Ranking, self).__init__(embedding_dim, Fiedler_layer_num, alpha, 
                trainable_alpha, initial_score, prob_dim, sigma, **kwargs)
        self.ib1 = InceptionBlock(num_features, embedding_dim)
        self.ib2 = InceptionBlock(embedding_dim, embedding_dim)
        self.ib3 = InceptionBlock(embedding_dim, prob_dim)
        self._dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, \
        edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor], \
        edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor], \
        features: torch.FloatTensor,) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN ranking model with inception blocks modified from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
            * features (PyTorch FloatTensor) - Node features.
            
        Return types:
            * x (PyTorch FloatTensor) - Class probabilities for all nodes, 
                with shape (num_nodes, prob_dim).
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        self.z = x.clone()
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2

        return F.softmax(x, dim=1)