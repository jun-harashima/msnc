import torch
import torch.nn as nn
import torch.nn.utils as U
from msnc.util import Util


class Encoder(nn.Module):

    def __init__(self, xdim, edim, hdim, lnum, use_bidirectional=True,
                 use_lstm=True, dropout=0.2):
        super(Encoder, self).__init__()
        self.util = Util()
        self.pad_index = self.util.PAD_INDEX
        self.xdim = xdim
        self.edim = edim
        self.hdim = hdim
        self.lnum = lnum
        self.use_bidirectional = use_bidirectional
        self.use_lstm = use_lstm
        self.dropout = dropout
        self.use_cuda = self.util.use_cuda
        self.embedding = self._init_embedding()
        self.rnn = self._init_rnn()

    def _init_embedding(self):
        embedding = nn.Embedding(self.xdim, self.edim, self.pad_index)
        return embedding.cuda() if self.use_cuda else embedding

    def _init_rnn(self):
        return self._init_lstm() if self.use_lstm else self._init_gru()

    def _init_lstm(self):
        lstm = nn.LSTM(self.edim, self.hdim, num_layers=self.lnum,
                       batch_first=True, bidirectional=self.use_bidirectional,
                       dropout=self.dropout)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_gru(self):
        gru = nn.GRU(self.edim, self.hdim, num_layers=self.lnum,
                     batch_first=True, bidirectional=self.use_bidirectional,
                     dropout=self.dropout)
        return gru.cuda() if self.use_cuda else gru

    def forward(self, X):
        X, indices_before_sort, lengths_after_sort = self._sort(X)
        X = self._pad(X, lengths_after_sort)
        X = self.util.tensorize(X)
        X = self._embed(X)
        X = self._pack(X, lengths_after_sort)
        H = self._rnn(X)
        H = self._cat(H) if self.use_bidirectional else self._view(H)
        H = self._unsort(H, indices_before_sort)
        return H

    def _sort(self, X):
        indices, X = zip(*sorted(enumerate(X), key=lambda x: -len(x[1])))
        return list(X), self.util.tensorize(list(indices)), [len(x) for x in X]

    def _pad(self, X, lengths):
        for i, x in enumerate(X):
            X[i] = x + [self.pad_index] * (max(lengths) - len(X[i]))
        return X

    def _embed(self, X):
        return self.embedding(X)

    def _pack(self, X, lengths):
        return U.rnn.pack_padded_sequence(X, lengths, batch_first=True)

    def _rnn(self, X):
        _, H = self.rnn(X)
        return H[0] if self.use_lstm else H

    def _cat(self, H):
        forward_H = H[-2, :, :]
        backward_H = H[-1, :, :]
        return torch.cat((forward_H, backward_H), 1)

    def _view(self, H):
        return H.view(-1, self.hdim)

    def _unsort(self, H, indices):
        _, unsorted_indices = torch.tensor(indices).sort()
        return H.index_select(0, unsorted_indices)
