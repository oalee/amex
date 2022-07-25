from torch import nn
from .conv1d.conv1d import GaussianNoise
import torch as t


class RoundFloat(nn.Module):
    def __init__(self, precision=1e4):
        super().__init__()
        self.precision = precision

    def forward(self, x):
        # round x to 0.0001 precision
        return t.round(x * self.precision) / self.precision
        return t.round(x * 10000) / 10000
        return t.round(x)


class TabularEmbedding(nn.Module):
    def __init__(self, params, embed_time=False):
        super().__init__()
        self.params = params
        # embedding dim = params.in_feature * h_embedding_dim
        self.h_embedding_dim = params.hparams.feature_embed_dim
        h_embedding = params.hparams.feature_embed_dim

        in_features = 157

        self.time_embedding = nn.Embedding(13, in_features * h_embedding)
        self.embed_time = embed_time

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
                # nn.Sequential(GaussianNoise(params.hparams.noise_std))
                nn.Sequential(RoundFloat(1e4))
                if h_embedding == 1
                else nn.Sequential(
                    GaussianNoise(params.hparams.noise_std), nn.Linear(1, h_embedding)
                )
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
        embeddings = t.stack(embeddings, dim=2)

        pos_emb = self.pos_emb(t.arange(D, device=x.device))
        pos_emb = pos_emb.repeat(B * T, 1, 1).flatten(1)
        # ipdb.set_trace()
        embeddings = embeddings.flatten(1)
        embeddings = embeddings + pos_emb
        embeddings = self.act(embeddings)

        embeddings = embeddings.view(B, T, -1)

        if self.embed_time:
            time_embeddings = self.time_embedding(t.arange(T, device=x.device)).expand(
                B, T, 157 * self.h_embedding_dim
            )
            embeddings = embeddings + time_embeddings

        return embeddings
