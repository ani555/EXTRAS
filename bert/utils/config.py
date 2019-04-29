import os
root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = "../nlu_proj/data/finished_files/chunked/train_*"
eval_data_path = "../nlu_proj/data/finished_files/val.bin"
decode_data_path = "../nlu_proj/data/finished_files/test.bin"
vocab_path = "../nlu_proj/data/finished_files/vocab"
log_root = "log"

# Hyperparameters
d_model=768
d_ff=1024
nheads=8
num_layers=6
drop_prob=0.1
batch_size=8
max_enc_steps=400
max_dec_steps=100
min_dec_steps=35
vocab_size=30524
beam_size=4
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
pointer_gen=True
coverage=False
cov_loss_wt=1.0
eps=1e-12
max_iters=200000
lr_coverage=0.15
