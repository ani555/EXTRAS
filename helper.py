from torch.autograd import Variable
import numpy as np
import torch
from utils import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_src_batch(batch, config):
  
  batch_size = len(batch.enc_lens)
  enc_batch = torch.from_numpy(batch.enc_batch).long().to(device)
  enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float().to(device=device)
  enc_padding_mask = enc_padding_mask.unsqueeze(1)
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None

  if config['pointer_gen']:
    enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long().to(device=device)

    if batch.max_art_oovs > 0:
      extra_zeros = torch.zeros((batch_size, config['max_dec_len'], batch.max_art_oovs)).to(device=device)

  # c_t_1 = torch.tensor(torch.zeros((batch_size, 2 * config.hidden_dim)), device=device)

  # coverage = None
  # if config['coverage']:
  #   coverage = torch.tensor(torch.zeros(enc_batch.size()), device=device)

  #   c_t_1 = c_t_1.cuda()

  #   if coverage is not None:
  #     coverage = coverage.cuda()

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros

def subsequent_mask(size):
  
  "Mask out subsequent positions."
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return (torch.from_numpy(subsequent_mask) == 0).to(device=device)

def prepare_tgt_batch(batch):
  dec_batch = torch.from_numpy(batch.dec_batch).long().to(device=device)
  dec_padding_mask = torch.from_numpy(batch.dec_padding_mask).int().to(device=device)
  # print('dec_padding_mask type:', dec_padding_mask.dtype)
  dec_padding_mask = dec_padding_mask.unsqueeze(1) & subsequent_mask(dec_batch.size(-1)).type_as(dec_padding_mask.data)
  # print(dec_padding_mask.shape)
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = torch.from_numpy(dec_lens).float().to(device=device)

  # print(batch.target_batch[batch.target_batch>=10000])

  target_batch = torch.from_numpy(batch.target_batch).long().to(device=device)

  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

