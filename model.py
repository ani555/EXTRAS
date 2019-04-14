import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import ScaledDotProductAttention, MultiHeadedAttention


class LayerNorm(nn.Module):
    
    def __init__(self, d_model, eps=1e-6):
        
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean)/(std + self.eps) + self.beta


class PositionWiseFeedForwardNetwork(nn.Module):
    
    def __init__(self, d_model, d_ff, drop_prob):
        
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.w1_layer = nn.Linear(d_model, d_ff)
        self.w2_layer = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        return self.w2_layer(self.dropout(F.relu(self.w1_layer(x))))



class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_ff, nheads, drop_prob=0.2):
        
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadedAttention(d_model, nheads, drop_prob)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff, drop_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x, mask=None):
        
        context, _ = self.attn_layer(x, x, x, mask)
        x = x + self.dropout(self.layer_norm1(context))

        ffn_output = self.ffn(x)
        x =  x + self.dropout(self.layer_norm2(ffn_output))
        
        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, d_ff, nheads, drop_prob=0.2):
        
        super(DecoderLayer, self).__init__()
        self.dec_attn_layer = MultiHeadedAttention(d_model, nheads, drop_prob)
        self.enc_dec_attn_layer = MultiHeadedAttention(d_model, nheads, drop_prob)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff, drop_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)
        
        
    def forward(self, x, enc_outputs, src_mask=None, tgt_mask=None):
        
        self_context, _ = self.dec_attn_layer(x, x, x, tgt_mask) # context from self attention on the decoder layers
        x = x + self.dropout(self.layer_norm1(self_context))

        cross_context, cross_attn_op = self.enc_dec_attn_layer(querys=x, keys=enc_outputs, values=enc_outputs, mask=src_mask) # context from attn on encoder states and decoder prev states
        x = x + self.dropout(self.layer_norm2(cross_context))
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(self.layer_norm3(ffn_output))
        
        return x

class Encoder(nn.Module):
    
    def __init__(self, d_model=512, d_ff=2048, nheads=8, num_layers=6, drop_prob=0.2):
        
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList([EncoderLayer(d_model, d_ff, nheads, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        
        for encoder in self.encoders:    
            x = encoder(x, mask)
        
        return x
        
class Decoder(nn.Module):
    
    def __init__(self, d_model=512, d_ff=2048, nheads=8, num_layers=6, drop_prob=0.2):
        
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList([DecoderLayer(d_model, d_ff, nheads, drop_prob) for _ in range(num_layers)])
        
    def forward(self, x, enc_outputs, src_mask=None, tgt_mask=None):
        
        for decoder in self.decoders:
            x = decoder(x, enc_outputs, src_mask, tgt_mask)
        
        return x

class PositionalEncoding(nn.Module):
    # have the same dimension as the embeddings coz we need to sum them
    # precompute and store the pos encodings
    def __init__(self, d_model, max_len, drop_prob=0.2):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        
        # compute the div_term in log space turns out to be efficient
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (- math.log(1000)/d_model))
        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)
        pe = pe.unsqueeze(0) # unsqueeze the batch dimension
        self.pos_enc = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x):
        return self.dropout(self.pos_enc[:,:x.size(1),:])
        
        
class InputEmbedding(nn.Module):
	
	def __init__(self, d_model, vocab_size, max_len, drop_prob=0.2):

 		super(InputEmbedding, self).__init__()
 		self.embeddings = nn.Embedding(vocab_size, d_model)
 		self.pos_enc = PositionalEncoding(d_model, max_len, drop_prob)
	
	def forward(self, x):
		return self.embeddings(x) + self.pos_enc(x)
        

class Generator(nn.Module):

	def __init__(self, d_model, vocab_size):
		super(Generator, self).__init__()
		self.fc = nn.Linear(d_model, vocab_size)

	def forward(self, x, enc_outputs=None, src_mask=None, enc_batch_extend_vocab=None, extra_zeros=None):
		return F.log_softmax(self.fc(x), dim=-1)


class PointerGenerator(nn.Module):

    def __init__(self, d_model, vocab_size, nheads):
        super(PointerGenerator, self).__init__()
        self.fc = nn.Linear(d_model, vocab_size)
        self.p_gen_fc = nn.Linear(d_model, 1)
        self.single_head_attn = MultiHeadedAttention(d_model, 1)
        #self.extra_zeros = extra_zeros
    
    def forward(self, x, enc_outputs, src_mask, enc_batch_extend_vocab, extra_zeros):
        p_vocab = F.softmax(self.fc(x), dim=-1)
        #cross_attn_op = cross_attn_op.permute(0,2,3,1).contiguous()
        #print(cross_attn_op.shape)
        cross_context, cross_attn_op = self.single_head_attn(querys=x, keys=enc_outputs, values=enc_outputs, mask=src_mask)
        attn_probs = cross_attn_op.squeeze(1)
        p_gen = torch.sigmoid(self.p_gen_fc(x)) #[BXTy]
        #print(p_gen.shape)
        # if enc_batch_extend_vocab is not None:
        #     print("enc_batch_extend_vocab: ",enc_batch_extend_vocab.shape)
        attn_dist = (1-p_gen)*attn_probs
        p_vocab = p_gen*p_vocab

        if extra_zeros is not None:
            p_vocab_extended = torch.cat((p_vocab, extra_zeros), dim=-1)
        else:
            p_vocab_extended = p_vocab


        indices = enc_batch_extend_vocab.expand(attn_dist.size(1),-1,-1).transpose(0,1)
        p_vocab_extended.scatter_add_(-1, indices, attn_dist)
        
        #print(enc_batch_extend_vocab.shape)
        # print("p_vocab_extended: ",p_vocab_extended.shape)
        return torch.log(torch.clamp(p_vocab_extended, min = 1e-9))

class Transformer(nn.Module):

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):

		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)


	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, enc_output, tgt, src_mask, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), enc_output, src_mask, tgt_mask), enc_output