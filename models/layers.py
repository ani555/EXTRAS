import torch.nn as nn
import torch
import torch.nn.functional as f
from models.attention import SelfAttention


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


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        result = f.log_softmax(self.proj(x), dim=-1)
        _, top_indices = result.data.topk(1)
        return result.view(1, -1), top_indices


class TransformerEncoderLayer(nn.Module):

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


class TransformerDecoderLayer(nn.Module):

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
            self.encoder.append(TransformerEncoderLayer(dim, attention_heads, dim_ff, dropout))

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
            self.decoder.append(TransformerDecoderLayer(dim, attention_heads, dim_ff, dropout))

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor):
        for i in range(self.N):
            decoder_input = self.decoder[i](decoder_input, encoder_output, mask=None)
        return decoder_input
