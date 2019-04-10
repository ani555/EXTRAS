import torch.nn as nn
import torch
import torch.nn.functional as f
from torch.autograd import Variable
import math


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

    def __init__(self, dim, max_len=400, vocabulary_size=50000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.mul(torch.arange(0, dim, 2), -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(torch.mul(position, div_term))
        pe[:, 1::2] = torch.cos(torch.mul(position, div_term))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.lut = nn.Embedding(vocabulary_size, dim)

    def forward(self, tensor):
        emb = self.lut(tensor) * math.sqrt(self.dim)
        result = emb + Variable(self.pe[:, :emb.size(1)], requires_grad=False)
        return self.dropout(result)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        result = f.log_softmax(self.proj(x), dim=-1)
        _, top_indices = result.data.topk(1)
        return result.view(1, -1), top_indices


class EncoderLayer(nn.Module):

    def __init__(self, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()

        # define multi head attention layer
        self.self_attn_layer = nn.ModuleList()
        for i in range(attention_heads):
            self.self_attn_layer.append(SelfAttention(dim, attention_heads))

        # define feed forward layer
        self.feed_forward_layer = FeedForward(dim, dim_ff, dropout)

        # define residual connections with normalization
        self.residual_connection = nn.ModuleList()
        for i in range(3):
            self.residual_connection.append(ResidualConnection(dropout))

    def forward(self, encoder_input: torch.Tensor):
        multi_head_self_attn = [attn_head(encoder_input, encoder_input, encoder_input, None) for attn_head in self.self_attn_layer]
        self_attn = torch.cat(tuple(multi_head_self_attn), dim=2)
        residual_self_attn = self.residual_connection[0](encoder_input, self_attn)

        feed_forward_result = self.feed_forward_layer(residual_self_attn)
        return self.residual_connection[1](residual_self_attn, feed_forward_result)


class DecoderLayer(nn.Module):

    def __init__(self, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()

        # define multi head attention1
        self.decoder_self_attn_layer = nn.ModuleList()
        for i in range(attention_heads):
            self.decoder_self_attn_layer.append(SelfAttention(dim, attention_heads))

        # define multi head attention2
        self.encoder_decoder_self_attn_layer = nn.ModuleList()
        for i in range(attention_heads):
            self.encoder_decoder_self_attn_layer.append(SelfAttention(dim, attention_heads))

        # define feed forward layer
        self.feed_forward_layer = FeedForward(dim, dim_ff, dropout)

        # define residual connections with normalization
        self.residual_connection = nn.ModuleList()
        for i in range(3):
            self.residual_connection.append(ResidualConnection(dropout))

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor):
        decoder_multi_head_self_attn = [attn_head(decoder_input, decoder_input, decoder_input, mask) for attn_head in self.decoder_self_attn_layer]
        decoder_self_attn = torch.cat(tuple(decoder_multi_head_self_attn), dim=2)
        residual_decoder_self_attn = self.residual_connection[0](decoder_input, decoder_self_attn)

        encoder_decoder_multi_head_self_attn = [attn_head(residual_decoder_self_attn, encoder_output, encoder_output, mask) for attn_head in self.encoder_decoder_self_attn_layer]
        encoder_decoder_self_attn = torch.cat(tuple(encoder_decoder_multi_head_self_attn), dim=2)
        residual_encoder_decoder_self_attn = self.residual_connection[1](residual_decoder_self_attn, encoder_decoder_self_attn)

        feed_forward_output = self.feed_forward_layer(residual_encoder_decoder_self_attn)
        return self.residual_connection[2](residual_encoder_decoder_self_attn, feed_forward_output)


class TransformerEncoder(nn.Module):
    def __init__(self, layers, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.N = layers
        self.encoder = nn.ModuleList()
        for i in range(layers):
            self.encoder.append(EncoderLayer(dim, attention_heads, dim_ff, dropout))

    def forward(self, input_tensor: torch.Tensor):
        for i in range(self.N):
            input_tensor = self.encoder[i](input_tensor)
        return input_tensor


class TransformerDecoder(nn.Module):
    def __init__(self, layers, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.N = layers
        self.decoder = nn.ModuleList()
        for i in range(layers):
            self.decoder.append(DecoderLayer(dim, attention_heads, dim_ff, dropout))

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor):
        for i in range(self.N):
            decoder_input = self.decoder[i](decoder_input, encoder_output, mask=None)
        return decoder_input
