import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import ast
from tqdm import tqdm
# import matplotlib.pyplot as plt

from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab
from model import BiDirectLSTM
    
# Model parameters
vocab_size = len(word_vocab)
output_size = len(gloss_vocab)
embedding_dim = 128
hidden_dim = 256
n_layers = 3

print("CUDA availability: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = BiDirectLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model = model.to(device)
###########
# Training
###########

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 7

for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    val_loss = 0.0
    
    model.train()
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    for i, (inputs, labels) in tqdm(enumerate(val_loader)):
        outputs = model(inputs)
        loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch: {epoch+1}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss}")

# # Plot train and validation losses
# 
# plt.plot(train_losses, label='Training loss')
# plt.plot(val_losses, label='Validation loss')
# plt.legend()
# plt.show()

# Save the model
torch.save(model.state_dict(), 'bi_direct_lstm2.pth')

