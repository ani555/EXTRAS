import torch.nn as nn
import torch
import torch.nn.functional as f


# implement mask
class SelfAttention(nn.Module):

    def __init__(self, dim, attention_heads):
        super().__init__()
        self.dim = dim
        self.query_layer = nn.Linear(dim, int(dim/attention_heads))
        self.key_layer = nn.Linear(dim, int(dim/attention_heads))
        self.valueLayer = nn.Linear(dim, int(dim/attention_heads))

    def forward(self, query_source, key_source, value_source):

        query: torch.Tensor = self.query_layer(query_source)
        key: torch.Tensor = self.key_layer(key_source)
        value: torch.Tensor = self.value_layer(value_source)

        _, query_seq_len, _ = query.size()
        _, key_seq_len, _ = key.size()

        # query: (batch, query_seq_len, dim)
        # key: (batch, key_seq_len, dim)
        # value: (batch, key_seq_len, dim)
        # repeat query

        query = query.view((-1, self.dim))\
            .repeat((1, key_seq_len))\
            .view((-1, self.dim))  # (batch * query_seq_len * key_seq_len, dim)

        key = key.repeat((1, query_seq_len, 1))\
            .view((-1, self.dim))  # (batch * key_seq_len * query_seq_len, dim)

        dot_product = torch.sum(torch.mul(query, key), 1)\
            .view((-1, key_seq_len))  # (batch * query_seq_len, key_seq_len)

        # do masking

        dist = f.softmax(dot_product, 1).view(-1, 1).repeat(1, self.dim)

        value = value.repeat((1, query_seq_len, 1)) \
            .view((-1, self.dim))  # (batch * key_seq_len * query_seq_len, dim)

        weighted_avg1 = torch.mul(value, dist)\
            .view((-1, key_seq_len, self.dim))

        weighted_avg2 = torch.sum(weighted_avg1, 1)\
            .view((-1, query_seq_len, self.dim))

        return weighted_avg2


# FFN(x)=max(0,xW1+b1)W2+b2
class FeedForward(nn.Module):

    def __init__(self, dim, dim_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim_ff)
        self.w2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        return self.w_2(self.dropout(f.relu(self.w_1(tensor))))


class ResidualConnection(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # recheck proper implementation of layer norm
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, source, target):
        return source + self.dropout(self.layer_norm(target))


class EmbeddingLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        pass


class PredictionLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        pass
