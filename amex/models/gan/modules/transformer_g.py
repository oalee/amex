import torch
from torch import nn
import torch.nn.functional as F
import ipdb


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


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        hparams = params.hparams

        in_features = hparams.in_features
        self.in_features = in_features
        embedding_dim = 188
        num_heads = hparams.num_heads
        depth = hparams.depth
        seq_length = 13
        self.embedding_dim = 188

        self.conditional_embedding = nn.Sequential(
            nn.Embedding(2, embedding_dim // 2),
        )

        z_dim = 64
        self.z_dim = z_dim
        self.noise_embedding = nn.Sequential(
            nn.Linear(z_dim, 13 * (embedding_dim // 2)), nn.ReLU()
        )

        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=seq_length
        )
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(TransformerBlock(embedding_dim, num_heads))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_predictions = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, y):

        # B, 13, 188 = x.shape

        # B, 1 = y.shape

        # y = y.unsqueeze(1)
        B, _ = y.shape

        embed_condition = self.conditional_embedding(y.int())
        # ipdb.set_trace()
        noise = torch.randn(B, self.z_dim, device=y.device)
        noise_embedding = self.noise_embedding(noise)
        noise_embedding = noise_embedding.view(B, 13, self.embedding_dim//2)
        embed_condition = embed_condition.repeat(1, 13, 1)
        # ipdb.set_trace()

        noise_embedding = torch.cat([embed_condition, noise_embedding], dim=-1)
        # x = torch.cat([noise_embedding, embed_condition], dim=-1)
        # ipdb.set_trace()
        # batch_size, tweet_length, embedding_dim = x.shape
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(13, device=y.device)), 0
        ).expand(B, 13, self.embedding_dim )

        x = noise_embedding + positions
        x = self.transformer_blocks(x)
        x = self.to_predictions(x)
        # x = self.to_probabilities(x)

        return x
