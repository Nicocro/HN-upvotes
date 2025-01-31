# relevant imports
import tqdm
import collections
import requests
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import W2VTokenizer
from cbow import CBOW, CBOWDataset, cbow_loss

# Initialize tokenizer
tokenizer = W2VTokenizer(min_freq=7, use_stemming=False)

# Load and preprocess text, create vocabulary
with open('text8.txt') as f: text8: str = f.read()
tokenizer.fit([text8])

#tokenize the text8 dataset (a long string)
tokens = tokenizer.tokenize(text8)

# set up hyperparameters
VOCAB_SIZE = len(tokenizer.vocab)
EMBEDDING_DIM = 64
WINDOW_SIZE = 2
BATCH_SIZE = 128
NUM_EPOCHS = 3

# instantiate stuff
dataset = CBOWDataset(tokens, tokenizer, WINDOW_SIZE, VOCAB_SIZE)
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# put things to device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# wandb configs
wandb.init(project='text8-embeds', 
           name='CBOW-Text8', 
           config={'vocab_size': VOCAB_SIZE, 
                   'embedding_dim': EMBEDDING_DIM, 
                   'window_size': WINDOW_SIZE, 
                   'batch_size': BATCH_SIZE, 
                   'num_epochs': NUM_EPOCHS,
                   'device': device.type,
                   'tokenizer_min_freq': tokenizer.min_freq,
                   'tokenizer_use_stemming': tokenizer.use_stemming,
                   'tokenizer_fit_dataset': "text8",
                   'embedding_fit_dataset': "text8"
                   })

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (context, target, neg_samples) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    ):
        context, target, neg_samples = context.to(device), target.to(device), neg_samples.to(device)
        optimizer.zero_grad()
        outputs = model(context) # forward pass
        loss = cbow_loss(outputs, target, neg_samples) #calcuiate loss
        
        # backward passs
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        wandb.log({'loss': loss.item()})

# Save model
torch.save(model.state_dict(), './cbow_text8_weights.pt')
artifact = wandb.Artifact('cbow-text8-model-weights', type='model')
artifact.add_file('./cbow_text8_weights.pt')
wandb.log_artifact(artifact)
wandb.finish()