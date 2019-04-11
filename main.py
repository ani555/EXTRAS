import tensorflow as tf
import torch
import torch.nn as nn
from utils.batcher import Batcher
from utils.data import Vocab
from helper import prepare_src_batch, prepare_tgt_batch
import json
import argparse
from model import *

parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--mode', dest='mode', default='train', help='train or test', required=True)
parser.add_argument('--vocab_path', dest='vocab_path', default='data/finished_files/vocab', help='location of the source file')
parser.add_argument('--train_data_path', dest='train_data_path', default='data/finished_files/chunked/*', help='location of the source file')
parser.add_argument('--config_file', dest='config_file', default='config.json', help='config file name with path')
parser.add_argument('--save_path', dest='save_path', default='ckpt/', help='Path where model will be saved')
parser.add_argument('--load_path', dest='load_path', help='Path where model is present')
args = parser.parse_args()


class LabelSmoothing(nn.Module):

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model, config):
    return NoamOpt(config['d_model'], 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()


def build_model(config):


	encoder = Encoder(config['d_model'], config['d_ff'], config['nheads'], config['num_layers'], config['drop_prob'])
	decoder = Decoder(config['d_model'], config['d_ff'], config['nheads'], config['num_layers'], config['drop_prob'])
	embedding = InputEmbedding(config['d_model'], config['vocab_size'], config['max_enc_len'], config['drop_prob'])
	generator = Generator(config['d_model'], config['vocab_size'])
	model = Transformer(encoder, decoder, embedding, embedding, generator)
	model.cuda()

	return model


def train_step(batch, model, config, loss_compute):

	model.train()
	enc_batch, enc_padding_mask, enc_lens = prepare_src_batch(batch, config)
	dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch =  prepare_tgt_batch(batch)
	# print(enc_batch.dtype)
	# print(dec_batch.dtype)
	out = model.forward(enc_batch, dec_batch, enc_padding_mask, dec_padding_mask)
	norm = int(torch.sum(dec_lens_var))
	loss = loss_compute(out, target_batch, norm)
	return loss




def train(max_iters, batcher, model, criterion, config):
	total_loss = 0.0
	opt = get_std_opt(model, config)
	loss_compute = SimpleLossCompute(model.generator, criterion, opt)
	batch_i=0
	while batch_i < max_iters:

		batch = batcher.next_batch()
		loss = train_step(batch, model, config, loss_compute)
		total_loss += loss
		if batch_i % 100 == 0:
			total_loss/=100
			print("Batch : {:>3} Avg Train Loss {:.5f}".format(batch_i, total_loss))
			total_loss = 0.0

		batch_i+=1

def main():

	with open('config.json', 'r') as f:
		config = json.load(f)
	vocab = Vocab(args.vocab_path, config['vocab_size'])
	batcher = Batcher(args.train_data_path, vocab, mode='train', batch_size=config['batch_size'], single_pass=False)
	model = build_model(config)
	criterion = LabelSmoothing(config['vocab_size'], batcher.pad_id, smoothing=.1)
	train(config['max_iters'], batcher, model, criterion, config)

if __name__ == '__main__':
	main()