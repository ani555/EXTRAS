from models.layers import *


class Model(nn.Module):
    def __init__(self, embedding, encoder, decoder, attention, prediction, loss, max_decode_length):
        super().__init__()
        self.embedding_layer = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.prediction = prediction
        self.loss = loss

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, inference=False, learn=False):
        loss = 0
        batch_size, _, _ = encoder_input.size()
        encoder_input = self.embedding_layer(encoder_input)
        encoder_output = self.encoder(encoder_input)

        prediction = torch.empty((1, batch_size), dtype=torch.long, device=torch.device('cuda'))  # (1, batch_size)
        prediction.fill_(2)

        if inference:
            max_len = self.max_decode_length
        else:
            _, max_len, _ = decoder_input.size()

        for i in range(max_len - 1):
            if inference:
                tensor = prediction[:, 0:i+1, :]  # should be picked from result
            else:
                tensor = decoder_input[:, 0:i+1, :]  # should be picked from input
            tensor = self.embedding_layer(tensor)
            decoder_output = self.decoder(encoder_output, tensor)

            context = None
            attn_dist = None
            if self.attention is not None:
                context, attn_dist = self.attention(decoder_output, encoder_output)

            dist, top_indices = self.prediction_layer(decoder_output, context, attn_dist)

            if not inference:
                loss_tensor = self.loss(dist, decoder_input[:, i+1])
                loss += float(loss_tensor)

                if learn:
                    loss_tensor.backward(retain_graph=True)

            prediction = torch.cat((prediction, top_indices))
            if attn_dist is not None:
                # get c_t(aggregate of previous attn results)
                pass

        return prediction, loss
