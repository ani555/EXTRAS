import tensorflow as tf
import torch
import torch.nn as nn
from utils import config as conf
from utils.batcher import Batcher
from utils.data import Vocab, VocabBert
from helper import prepare_src_batch, prepare_tgt_batch
import json
import argparse
from model import *
from beam_search import BeamSearchDecoder
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import config
torch.cuda.set_device(5)

parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--mode', dest='mode', default='train', help='train or eval', required=True)
parser.add_argument('--save_path', dest='save_path', default='ckpt/', help='Path where model will be saved')
parser.add_argument('--model_file', dest='model_file', help='Path where model is present')
args = parser.parse_args()

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert_enc = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def forward(self, encoder_batch, encoder_padding_mask):
        return self.bert_enc(encoder_batch, attention_mask = encoder_padding_mask, output_all_encoded_layers=False)


    def tokenize(self, tokenized_sen):
        return [self.tokenizer.vocab[key] if key in self.tokenizer.vocab else self.tokenizer.vocab['[UNK]'] for key in tokenized_sen]

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
    return NoamOpt(config.d_model, 1, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm, enc_op, enc_padding_mask, enc_batch_extend_vocab, extra_zeros):

        if config.coverage:
            x, cov_loss = self.generator(x, enc_op, enc_padding_mask, enc_batch_extend_vocab, extra_zeros)
            loss = (self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) + config.cov_loss_wt*cov_loss) / norm
        else:
            x = self.generator(x, enc_op, enc_padding_mask, enc_batch_extend_vocab, extra_zeros)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()

def eval(config):

    model = build_model_bert(config)
    model.load_state_dict(torch.load(args.model_file))
    dec = BeamSearchDecoder(model)
    dec.decode(config)   



def build_model_bert(config):

    encoder = Bert()
    decoder = Decoder(config.d_model, config.d_ff, config.nheads, config.num_layers, config.drop_prob)
    src_embedding = InputEmbedding(config.d_model, config.vocab_size, config.max_enc_steps, config.drop_prob)
    tgt_embedding = InputEmbedding(config.d_model, config.vocab_size, config.max_enc_steps, config.drop_prob)
    if config.pointer_gen:
        if config.coverage:
            generator = PointerGeneratorWithCoverage(config.d_model, config.vocab_size, config.nheads)
        else:
            generator = PointerGenerator(config.d_model, config.vocab_size, config.nheads)
    else:
        generator = Generator(config.d_model, config.vocab_size)

    model = Transformer(encoder, decoder, src_embedding, tgt_embedding, generator)
    model.cuda()

    return model


def build_model(config):


	encoder = Encoder(config.d_model, config.d_ff, config.nheads, config.num_layers, config.drop_prob)
	decoder = Decoder(config.d_model, config.d_ff, config.nheads, config.num_layers, config.drop_prob)
	src_embedding = InputEmbedding(config.d_model, config.vocab_size, config.max_enc_steps, config.drop_prob)
	tgt_embedding = InputEmbedding(config.d_model, config.vocab_size, config.max_enc_steps, config.drop_prob)
	if config.pointer_gen:
		if config.coverage:
			generator = PointerGeneratorWithCoverage(config.d_model, config.vocab_size, config.nheads)
		else:
			generator = PointerGenerator(config.d_model, config.vocab_size, config.nheads)
	else:
		generator = Generator(config.d_model, config.vocab_size)
	model = Transformer(encoder, decoder, src_embedding, tgt_embedding, generator)
	model.cuda()
	return model

def train_step_bert(batch, model, config, loss_compute):

    model.train()
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros = prepare_src_batch(batch)
    dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch =  prepare_tgt_batch(batch)

    model.encoder.eval()
    with torch.no_grad():
        enc_op, _ = model.encoder.forward(enc_batch, enc_padding_mask.squeeze(1))
    out, _ = model.decode(enc_op, dec_batch, enc_padding_mask, dec_padding_mask)
    norm = int(torch.sum(dec_lens_var))
    loss = loss_compute(out, target_batch, norm, enc_op, enc_padding_mask, enc_batch_extend_vocab, extra_zeros)  
    return loss    

def train_step(batch, model, config, loss_compute):

	model.train()
	enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros = prepare_src_batch(batch, config)
	dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch =  prepare_tgt_batch(batch)
	out, enc_op = model.forward(enc_batch, dec_batch, enc_padding_mask, dec_padding_mask)
	norm = int(torch.sum(dec_lens_var))
	loss = loss_compute(out, target_batch, norm, enc_op, enc_padding_mask, enc_batch_extend_vocab, extra_zeros)
	return loss




def train(max_iters, batcher, model, criterion, config, save_path):
    total_loss = 0.0
    avg_cov_loss = 0.0
    opt = get_std_opt(model, config)
    loss_compute = SimpleLossCompute(model.generator, criterion, opt)
    batch_i=0
    while batch_i < max_iters:

        batch = batcher.next_batch()
        loss = train_step_bert(batch, model, config, loss_compute)
        total_loss += loss
        if batch_i % 100 == 0:
            total_loss/=100
            print("Batch : {:>3} Avg Train Loss {:.5f}".format(batch_i, total_loss))
            total_loss = 0.0
        batch_i+=1
        if batch_i % 50000 == 0:
            file_name = "model"+str(batch_i)+".pt"
            torch.save(model.state_dict(),os.path.join(save_path, file_name))


def main():

    vocab = VocabBert(config.vocab_path, config.vocab_size)
    if args.mode == 'train':
        batcher = Batcher(config.train_data_path, vocab, mode='train', batch_size=config.batch_size, single_pass=False)
    elif args.mode == 'eval':
        batcher = Batcher(config.eval_data_path, vocab, mode='decode', batch_size=config.batch_size, single_pass=True)
    model = build_model_bert(config)
    criterion = LabelSmoothing(config.vocab_size, batcher.pad_id, smoothing=.1)
    
    if args.mode=='train':
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        train(config.max_iters, batcher, model, criterion, config, args.save_path)
    elif args.mode=='eval':
        eval(config)

if __name__ == '__main__':
	main()
