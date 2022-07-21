from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import ipdb
import torch as t

from .conv1d.conv1d import GaussianNoise, Conv1DLayers


class TabularEmbedding(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # embedding dim = params.in_feature * h_embedding_dim
        self.h_embedding_dim = params.hparams.feature_embed_dim
        h_embedding = params.hparams.feature_embed_dim

        in_features = 157

        embedding_type_dict = {}
        for i in range(4):
            embedding_type_dict[i] = nn.Embedding(2, h_embedding)

        for i in range(4, 6):
            embedding_type_dict[i] = nn.Embedding(3, h_embedding)

        for i in range(6, 7):
            embedding_type_dict[i] = nn.Embedding(4, h_embedding)

        for i in range(7, 8):
            embedding_type_dict[i] = nn.Embedding(6, h_embedding)

        for i in range(8, 11):
            embedding_type_dict[i] = nn.Embedding(7, h_embedding)

        for i in range(11, in_features):
            embedding_type_dict[i] = (
                nn.Sequential(GaussianNoise(0.001))
                if h_embedding == 1
                else nn.Sequential(GaussianNoise(0.001), nn.Linear(1, h_embedding))
            )

        self.embeddings = nn.ModuleList(list(embedding_type_dict.values()))

        self.na_embedding = nn.Embedding(1, h_embedding)

        self.act = nn.GELU()

        self.pos_emb = nn.Embedding(params.hparams.in_features, h_embedding)

    def __embedd_feature(self, x: t.Tensor, feature: int):
        (B,) = x.size()

        # find nans over batch
        nan_mask = t.isnan(x)
        nan_count = nan_mask.sum()

        output = t.zeros(B, self.h_embedding_dim, device=x.device)
        if nan_count > 0:
            # if there are nans, we need to replace them nan embeddings
            nan_embedding = self.na_embedding(t.zeros(nan_count, device=x.device).int())
            output[nan_mask] = nan_embedding

        if feature < 11:
            # input is int for embedding
            x = x.int()
            output[~nan_mask] = self.embeddings[feature](x[~nan_mask])
        else:
            x = x.unsqueeze(-1)
            output[~nan_mask] = self.embeddings[feature](x[~nan_mask])

        return output

    def forward(self, x: t.Tensor):
        # ipdb.set_trace()
        B, T, D = x.size()
        x = x.view(B * T, D)

        embeddings = [self.__embedd_feature(x[:, i], i) for i in range(D)]
        embeddings = torch.stack(embeddings, dim=2)

        pos_emb = self.pos_emb(t.arange(D, device=x.device))
        pos_emb = pos_emb.repeat(B * T, 1, 1).flatten(1)
        # ipdb.set_trace()
        embeddings = embeddings.flatten(1)
        embeddings = embeddings + pos_emb
        embeddings = self.act(embeddings)

        embeddings = embeddings.view(B, T, D, self.h_embedding_dim)
        return embeddings


from axial_attention import AxialAttention


class AxialLayers(nn.Module):
    def __init__(self, embedding_dim, num_layers, dropout, residual=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.attention = AxialAttention(embedding_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    AxialAttention(embedding_dim, 2),
                    nn.Dropout(dropout),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            x_ = self.layers[i](x)
            if self.residual:
                x = x_ + x
            else:
                x = x_
        return x


class AxialClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        hparams = params.hparams

        in_features = hparams.in_features
        embedding_dim = hparams.feature_embed_dim
        num_heads = 16
        depth = 6
        seq_length = 13
        dropout = 0.2
        # self.z_dim = 256 - 188

        self.embedding = TabularEmbedding(params)

        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim * in_features, num_embeddings=seq_length
        )

        self.to_probabilities = nn.Sequential(nn.Flatten(), nn.Linear(13 * 157, 1))

        self.axial_layers = AxialLayers(
            embedding_dim=embedding_dim,
            num_layers=12,
            dropout=dropout,
            residual=True,
        )

        self.noise = GaussianNoise(0.0001)

    def forward(self, x):
        # size x: (batch_size, T, D)

        if self.training:
            rand = t.rand_like(x, device=x.device)
            nan_mask = (
                rand < self.params.hparams.nan_prob * t.rand(1, device=x.device)[0]
            )

            x[nan_mask] = t.nan

        x = self.embedding(x)

        x = self.axial_layers(x)

        x = x  # + c_conv1d
        x = t.max(x, dim=-1)[0]
        x = self.to_probabilities(x)

        return x
