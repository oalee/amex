from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import ipdb
import torch as t

from .conv1d.conv1d import GaussianNoise, Conv1DLayers


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.to_keys = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_queries = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_values = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.unify_heads = nn.Linear(heads * embedding_dim, embedding_dim)

    def forward(self, x):
        # ipdb.set_trace()
        batch_size, tweet_length, embedding_dim = x.size()
        keys = self.to_keys(x).view(batch_size, tweet_length, self.heads, embedding_dim)
        queries = self.to_queries(x).view(
            batch_size, tweet_length, self.heads, embedding_dim
        )
        values = self.to_values(x).view(
            batch_size, tweet_length, self.heads, embedding_dim
        )
        keys = (
            keys.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = (
            queries.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        values = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = queries / (embedding_dim ** (1 / 4))
        keys = keys / (embedding_dim ** (1 / 4))

        dot = F.softmax(torch.bmm(queries, keys.transpose(1, 2)), dim=2)

        out = torch.bmm(dot, values).view(
            batch_size, self.heads, tweet_length, embedding_dim
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, tweet_length, self.heads * embedding_dim)
        )
        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, fc_hidden_multiply=4, dropout=0.02):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * fc_hidden_multiply),
            nn.ReLU(),
            nn.Linear(embedding_dim * fc_hidden_multiply, embedding_dim),
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)  # TODO: explain skip connection
        # x = self.do(x)

        feedforward = self.fc(x)
        x = self.norm2(feedforward + x)

        if self.training:
            x = self.do(x)
        # x = self.do(x)
        return x


class TabularEmbeddingDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.h_emb_dim = params.hparams.feature_embed_dim
        h_emb_dim = self.h_emb_dim
        in_features = params.hparams.in_features
        self.in_features = in_features

        embedding_decoders = {}
        for i in range(4):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 2)
        for i in range(4, 6):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 3)
        for i in range(6, 7):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 4)
        for i in range(7, 8):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 6)
        for i in range(8, 11):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 7)
        for i in range(11, in_features):
            embedding_decoders[i] = nn.Linear(h_emb_dim, 2)  # Mean and std

        self.embedding_decoders = nn.ModuleList(list(embedding_decoders.values()))

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, self.in_features, self.h_emb_dim)
        # ipdb.set_trace()
        outs = []
        for i in range(11):
            dec = self.embedding_decoders[i](x[:, i])
            dec = t.softmax(dec, dim=1)
            # convert softmax to int index
            dec = t.argmax(dec, dim=1)
            outs.append(dec)

        for i in range(11, self.params.hparams.in_features):
            dec = self.embedding_decoders[i](x[:, i])
            mean = t.tanh(dec[:, 0])
            std = t.sigmoid(dec[:, 1])
            # sample from normal distribution
            dec = t.normal(mean, std)
            # ipdb.set_trace()
            outs.append(dec)

        outs = t.stack(outs, dim=1)
        outs = outs.view(B, T, -1)
        return outs


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
                nn.Sequential(GaussianNoise(0.01))
                if h_embedding == 1
                else nn.Sequential(GaussianNoise(0.01), nn.Linear(1, h_embedding))
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

        embeddings = embeddings.view(B, T, -1)
        return embeddings


class TransformerClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        hparams = params.hparams

        in_features = hparams.in_features
        embedding_dim = in_features * hparams.feature_embed_dim
        num_heads = 16
        depth = 6
        seq_length = 13
        dropout = 0.3

        self.embedding = TabularEmbedding(params)

        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=seq_length
        )
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                TransformerBlock(embedding_dim, num_heads, dropout=dropout)
            )

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probabilities = nn.Linear(embedding_dim, 1)

        # self.conv1d = Conv1DLayers(5, 13, embedding_dim, dropout=0.2)

        self.noise = GaussianNoise(0.0001)

    def forward(self, x):
        # size x: (batch_size, T, D)

        if self.training:
            rand = t.rand_like(x, device=x.device)
            nan_mask = (
                rand < self.params.hparams.nan_prob * t.rand(1, device=x.device)[0]
            )
            # ipdb.set_trace()
            x[nan_mask] = t.nan

        x = self.embedding(x)

        x = x

        batch_size, tweet_length, embedding_dim = x.shape
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(tweet_length, device=x.device)), 0
        ).expand(batch_size, tweet_length, embedding_dim)

        x = x + positions
        x = self.transformer_blocks(x)
        x = x.max(dim=1)[0]

        x = x  # + c_conv1d
        x = self.to_probabilities(x)

        return F.sigmoid(x)


class TransformerDiscriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        hparams = params.hparams

        in_features = hparams.in_features
        embedding_dim = in_features * hparams.feature_embed_dim
        num_heads = 16
        depth = 6
        seq_length = 13
        dropout = 0.3

        self.cond_embedding = nn.Embedding(2, embedding_dim)
        self.embedding = TabularEmbedding(params)

        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=seq_length
        )
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                TransformerBlock(embedding_dim, num_heads, dropout=dropout)
            )

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probabilities = nn.Linear(embedding_dim, 1)

        # self.conv1d = Conv1DLayers(5, 13, embedding_dim, dropout=0.2)

        self.noise = GaussianNoise(0.0001)

    def forward(self, x, y):
        # size x: (batch_size, T, D)

        if y.dim() == 1:
            y = y.unsqueeze(1)
        B, _ = y.shape
        y = y.round().int()

        if self.training:
            rand = t.rand_like(x, device=x.device)
            nan_mask = (
                rand < self.params.hparams.nan_prob * t.rand(1, device=x.device)[0]
            )
            # ipdb.set_trace()
            x[nan_mask] = t.nan

        x = self.embedding(x)
        cond_emb = self.cond_embedding(y)

        # ipdb.set_trace()
        # ipdb.set_trace()
        cond_emb = cond_emb.repeat(1, 13, 1)
        x = x + cond_emb

        batch_size, tweet_length, embedding_dim = x.shape
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(tweet_length, device=x.device)), 0
        ).expand(batch_size, tweet_length, embedding_dim)

        x = x + positions
        x = self.transformer_blocks(x)
        x = x.max(dim=1)[0]

        x = x  # + c_conv1d
        x = self.to_probabilities(x)

        return F.sigmoid(x)


class TransformerGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        hparams = params.hparams

        in_features = hparams.in_features
        self.in_features = in_features
        embedding_dim = in_features * hparams.feature_embed_dim
        self.embedding_dim = embedding_dim
        self.h_emb_dim = hparams.feature_embed_dim
        num_heads = hparams.num_heads
        depth = hparams.depth

        self.feature_embedding = nn.Sequential(
            nn.Embedding(in_features, hparams.feature_embed_dim), nn.GELU()
        )

        self.dim_embedding = nn.Sequential(nn.Embedding(in_features, 1), nn.GELU())

        self.conditional_embedding = nn.Sequential(
            nn.Embedding(2, embedding_dim), nn.GELU()
        )

        z_dim = 64
        self.z_dim = z_dim
        self.noise_embedding = nn.Sequential(
            nn.Linear(z_dim, 13 * (embedding_dim)), nn.GELU()
        )

        self.positional_embedding = nn.Embedding(13, embedding_dim)

        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(TransformerBlock(embedding_dim, num_heads))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.decoder = TabularEmbeddingDecoder(params)

    def __embed_features(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D, 1)

        dim_emb = self.dim_embedding(t.arange(self.in_features, device=x.device))
        # ipdb.set_trace()
        dim_emb = dim_emb.repeat(B * T, self.h_emb_dim, 1)

        x = x + dim_emb

        x = x.view(B * T, self.in_features, self.h_emb_dim)
        feature_emb = self.feature_embedding(
            t.arange(self.in_features, device=x.device)
        )
        feature_emb = feature_emb.repeat(B * T, 1, 1)
        x = x + feature_emb

        x = x.view(B, T, D)
        return x

    def forward(self, y):

        # B, 13, 188 = x.shape

        # B, 1 = y.shape

        if y.dim() == 1:
            y = y.unsqueeze(1)
        B, _ = y.shape

        embed_condition = self.conditional_embedding(y.int())
        # ipdb.set_trace()
        noise = torch.randn(B, self.z_dim, device=y.device)
        noise_embedding = self.noise_embedding(noise)
        noise_embedding = noise_embedding.view(B, 13, self.embedding_dim)

        x = self.__embed_features(noise_embedding)

        embed_condition = embed_condition.repeat(1, 13, 1)
        # ipdb.set_trace()

        x = embed_condition + x
        # x = torch.cat([noise_embedding, embed_condition], dim=-1)
        # ipdb.set_trace()
        # batch_size, tweet_length, embedding_dim = x.shape
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(13, device=y.device)), 0
        ).expand(B, 13, self.embedding_dim)

        x = x + positions
        x = self.transformer_blocks(x)
        x = self.decoder(x)
        # ipdb.set_trace()
        # x = self.to_probabilities(x)

        return x
