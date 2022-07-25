import torch
from torch import nn
import ipdb
from .embed import TabularEmbedding
import torch as t


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.embedding = TabularEmbedding(params, embed_time=True)
        self.rnn = nn.LSTM(
            params.in_features * params.hparams.feature_embed_dim,
            params.hparams.h_dim,
            params.hparams.num_layers,
            batch_first=True,
            dropout=params.hparams.dropout,
        )

        self.fc1 = nn.Linear(params.hparams.h_dim, 1)
        # self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        # size x: (batch_size, T, D)
        if self.training:
            rand = t.rand_like(x, device=x.device)
            nan_mask = (
                rand < self.params.hparams.nan_prob * t.rand(1, device=x.device)[0]
            )
            # ipdb.set_trace()
            x[nan_mask] = t.nan

            # with 10% probability, replace a random T-dimensional vector with NaN
            prop = t.rand(1, device=x.device)[0]
            if prop < self.params.hparams.nan_time_prob:
                idx = t.randint(0, x.shape[1], (1,), device=x.device)[0]
                x[:, idx, :] = t.nan

        x = self.embedding(x)
        x, _ = self.rnn(x)
        # ipdb.set_trace()
        x = self.fc1(x[:, -1])

        return x

    def init_hidden(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0
