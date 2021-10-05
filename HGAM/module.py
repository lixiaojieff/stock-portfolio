import torch
import torch.nn as nn
import torch.nn.functional as F
from HGAM.layers import Attn_head_adj



class HGAT(nn.Module):
    def __init__(self, nfeat, nhid,  dropout):
        ''' attention mechanism. '''
        super(HGAT, self).__init__()
        self.dropout = dropout

        self.attention = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                         residual=True)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.attention(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        return x


