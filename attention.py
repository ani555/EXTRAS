import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, drop_prob=0.2):
        
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
    
    
    def forward(self, querys, keys, values, mask=None):
        
        # querys, keys, values - [BxnheadsxTxP]
        dk = keys.size(-1)
        attn_scores = torch.matmul(querys, torch.transpose(keys, 2, 3))/math.sqrt(dk)

        # print(mask.shape)
        # print(attn_scores.shape)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_outputs = F.softmax(attn_scores, dim=-1)
        attn_outputs = self.dropout(attn_outputs)
        context = torch.matmul(attn_outputs, values)
        
#         print('context: {}, attention outputs: {}'.format(context.shape, attn_outputs.shape))
        return context, attn_outputs
        
        
class MultiHeadedAttention(nn.Module):
    
    def __init__(self, d_model, nheads, drop_prob):
        
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.d_k = d_model//nheads # d_k = d_v = d_q
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.val_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(drop_prob)
    

    def forward(self, querys, keys, values, mask=None):
        
        batch_size = querys.size(0)
        # project and split the querys keys and values into nheads each of dimension [BxTxP], P=d_k
        proj_querys = self.query_layer(querys).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2) # [BxnheadsxTxP]
        proj_keys = self.key_layer(keys).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2) # [BxnheadsxTxP]
        proj_values = self.val_layer(values).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2) # [BxnheadsxTxP]
        
        
        context , attn_outputs = self.attn(proj_querys, proj_keys, proj_values, mask) #context [BxnheadsxTxd_model] attn_outputs [BxnheadsxTxT]
        context = torch.transpose(context, 1,2).contiguous().view(batch_size, -1, self.nheads*self.d_k) #[BxTxd_model]
        
        return self.output_layer(context), attn_outputs
