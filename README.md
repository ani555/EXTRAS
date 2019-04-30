# EXTRAS
Exploring Transformer Models for Abstractive Text Summarization

## About
This repository contains the implementation of transformer based models for abstractive text summarization. Here we have tried to integrate the pointer generator mechanism [1] with the transformer models for abstractive text summarization. The code is divided into two folders 
1. `transformer`- this folder contains the code for training and evaluating the transformer model with/without pointer generator
2. `bert` - this folder contains the code for BERT based model, in which we replace our original encoder with the pretrained BERT encoder. This model can also be trained and evaluated with/without pointer generator.

## Setup Instructions

### Setting up the environment
Python 3.5 and above is recommended to the run the code in this repo.

To install the requirements
```
pip3 install -r requirements.txt
```

### Dataset

Steps to obtain and process the CNN/DM dataset can be found [here](https://github.com/abisee/cnn-dailymail). Update `utils/config.py` with right dataset paths.

### Setting the hyperparameters
All the hyperparameters are loaded from `utils/config.py` file. Please set your hyperparameters in the same file.

### How to run

Run instructions are same for both models present in folders `transformer` and `bert`.

To train your model run:
```
python main.py --mode train --save_path ckpt/
```
To evaluate run:
```
python main.py --mode eval --model_file ckpt/model50000.pt
```


## References
<cite>[1] Abigail See, Peter J. Liu, and Christopher D. Manning.2017. Get to the point: Summarization with pointer-generator networks.Proceedings of the 55th AnnualMeeting of the Association for Computational Lin-guistics (Volume 1: Long Papers).</cite> <br>
<cite>[2] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.</cite><br>
<cite>[3]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep
bidirectional transformers for language understanding.</cite>
