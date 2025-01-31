import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import torch.nn as nn

from tokenizer import W2VTokenizer
from cbow import CBOW, CBOWDataset, cbow_loss

# Create a dummy dataset
dummy_data = "deep learning is amazing natural language processing is fun, machine learning is powerful artificial intelligence is the future"

# Initialize tokenizer
tokenizer = W2VTokenizer(min_freq=1, use_stemming=False)

# Fit tokenizer on dummy data
tokenizer.fit(dummy_data)

# Tokenize dummy data
tokens = tokenizer.tokenize(dummy_data)

# Create CBOW dataset
VOCAB_SIZE = len(tokenizer.vocab)
EMBEDDING_DIM = 64
WINDOW_SIZE = 2
BATCH_SIZE = 32
NUM_EPOCHS = 5

dataset = CBOWDataset(tokens, tokenizer, WINDOW_SIZE, VOCAB_SIZE)
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# put things to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
wandb.init(project='dummy-cbow', name='CBOW-Dummy')

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (context, target, neg_samples) in enumerate(dataloader):
        context, target, neg_samples = context.to(device), target.to(device), neg_samples.to(device)
        optimizer.zero_grad()
        outputs = model(context) # forward pass
        loss = cbow_loss(outputs, target, neg_samples) #calcuiate loss
        
        # backward passs
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    wandb.log({'loss': running_loss / len(dataloader)})

# Save model
torch.save(model.state_dict(), './cbow_dummy_weights.pt')
artifact = wandb.Artifact('cbow-dummy-model-weights', type='model')
artifact.add_file('./cbow_dummy_weights.pt')
wandb.log_artifact(artifact)
wandb.finish()