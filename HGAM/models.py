''' Define the HGAM  '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from HGAM.module import HGAT
import numpy as np


class HGAM(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            rnn_unit, n_hid,
            feature,
            dropout,
            ):

        super().__init__()

        # self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.LSTM(feature,
                          rnn_unit,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=False)

        self.HGAT = HGAT(nfeat=rnn_unit,
                         nhid=n_hid,
                         dropout=dropout,
                         )
        self.linear_out = nn.Linear(in_features=1 + n_hid, out_features=1)
        self.linear_out2 = nn.Linear(in_features=1 + n_hid, out_features=1)



    def forward(self, src_seq, adj, previous_w,  n_hid):
        # src_seq = self.linear(src_seq)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))

        rnn_output, *_ = self.rnn(src_seq)

        rnn_output = rnn_output[:, -1, :]
        enc_output = torch.reshape(rnn_output, (batch, stock, -1))
        HGAT_output = self.HGAT(enc_output, adj)

        HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))

        HGAT_output = torch.reshape(HGAT_output, (batch, stock, -1))

        previous_w = previous_w.permute(0, 2, 1)
        out = torch.cat([HGAT_output.cuda(), previous_w.cuda()], 2)

        out2 = self.linear_out2(out)
        out = self.linear_out(out)

        out = out.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)

        out = F.softmax(out, dim=-1)
        out2 = F.softmax(out2, dim=-1)

        out = out * 2
        out2 = -out2
        final_out = out + out2

        return final_out


