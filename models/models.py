import torch.nn as nn
import torch
from models.layers import SelfAttention
from models.layers import FeedForward
from models.layers import ResidualConnection


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


# add embedding layer
class TransformerEncoder(nn.Module):
    def __init__(self, layers, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.N = layers
        self.encoder = nn.ModuleList()
        for i in range(layers):
            self.encoder.append(EncoderLayer(dim, attention_heads, dim_ff, dropout))

    def forward(self, input_tensor: torch.Tensor):
        result = None
        for i in range(self.N):
            input_tensor = self.encoder[i](input_tensor)
            if result is None:
                result = input_tensor.clone()
            else:
                result = torch.cat((result, input_tensor))
        return result


class TransformerDecoder(nn.Module):
    def __init__(self, layers, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.N = layers
        self.decoder = nn.ModuleList()
        for i in range(layers):
            self.decoder.append(DecoderLayer(dim, attention_heads, dim_ff, dropout))

    def forward(self, encoder_outputs: torch.Tensor, decoder_input: torch.Tensor):
        for i in range(self.N):
            decoder_input = self.decoder[i](decoder_input, encoder_outputs[i], mask=None)
        return decoder_input


# add embedding layer
# add prediction layer
class Transformer(nn.Module):
    def __init__(self, layers, dim, attention_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.N = layers
        # initialize embedding layer
        self.t_encoder = TransformerEncoder(layers, dim, attention_heads, dim_ff, dropout)
        self.t_decoder = TransformerDecoder(layers, dim, attention_heads, dim_ff, dropout)
        # initialize prediction layer

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, inference=False):
        # pass encoder input through embedding layer
        encoder_output = self.t_encoder(encoder_input)
        prediction = None
        batch_size, decoder_seq_length, _ = decoder_input.size()
        for i in range(decoder_seq_length):
            if inference:
                tensor = 3  # should be picked from result
            else:
                tensor = 3  # should be picked from input
            # pass tensor through embedding layer
            decoder_output = self.t_decoder(encoder_output, tensor)

            # predict the result using prediction layer
            # compute loss

            # append prediction to result
        return prediction  # and error
