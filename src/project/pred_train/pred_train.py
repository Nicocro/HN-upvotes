import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# All absolute imports
from project.pred_train.pred_dataset import HNDataset
from project.repr_train.tokenizer import W2VTokenizer

tokenizer = W2VTokenizer()
dataset = HNDataset(csv_file='hn_corpus.csv', vocab_file='text8_vocab.json', model_file='cbow_text8_weights.pt', tokenizer=tokenizer, embedding_dim=64)
tokenizer.vocab = dataset.vocab

print(f'Dataset length: {len(dataset)}')

sample = dataset[0]
print(f'Sample embedding shape: {sample[0].shape}, Sample score: {sample[1]}')