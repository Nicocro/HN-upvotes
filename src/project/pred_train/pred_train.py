import pandas as pd
import numpy as np
import tqdm
import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# All absolute imports
from project.pred_train.pred_dataset import HNDataset
from project.repr_train.tokenizer import W2VTokenizer
from project.pred_train.pred_model import HNRegression

# Initialize Weights and Biases
wandb.init(project="hn-upvotes-regression")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# initialize stuff 
tokenizer = W2VTokenizer()
dataset = HNDataset(csv_file='hn_corpus.csv', vocab_file='text8_vocab.json', model_file='cbow_text8_weights.pt', tokenizer=tokenizer, embedding_dim=64)
tokenizer.vocab = dataset.vocab
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# some checks
print(f'Dataset length: {len(dataset)}')
sample = dataset[0]
print(f'Sample embedding shape: {sample[0].shape}, Sample score: {sample[1]}')
print(len(sample))

# define model, loss and optimizer 
model = HNRegression(embedding_dim=64, hidden_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Log model configuration to wandb
wandb.watch(model, log="all")

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(dataloader))
    
    avg_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
    
    # Log the average loss to wandb
    wandb.log({"epoch": epoch+1, "loss": avg_loss})

# Finish the wandb run
wandb.finish()