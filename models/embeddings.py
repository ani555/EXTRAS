import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class PositionalEmbedding(nn.Module):

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