import pandas as pd
import torch
import json 
from torch.utils.data import Dataset
from typing import List, Tuple
from project.repr_train.tokenizer import W2VTokenizer
from project.repr_train.cbow import CBOW

class HNDataset(Dataset):
    def __init__(self, csv_file: str, vocab_file: str, model_file: str, tokenizer: W2VTokenizer, embedding_dim: int):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[['title', 'score']].dropna()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim

        self.vocab = self.load_vocab(vocab_file)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.model = self.load_model(model_file)
        

    def load_vocab(self, vocab_file: str) -> dict:
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def load_model(self, model_file: str) -> torch.nn.Module:
        model = CBOW(len(self.vocab), self.embedding_dim)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()
        return model

    def __len__(self) -> int: 
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        title = self.data.iloc[idx]['title']
        score = self.data.iloc[idx]['score']
        tokens = self.tokenizer.tokenize(title)
        embeddings = [self.model.embeddings(torch.tensor(token)) for token in tokens]
        avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
        return avg_embedding, torch.tensor(score, dtype=torch.float32)