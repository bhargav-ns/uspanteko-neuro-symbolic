# Evaluate the model
# Find the F1 and accuracy scores for the model
import json
from sklearn.metrics import f1_score, accuracy_score
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

from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab
from load_data import indices_to_tokens
from model import BiDirectLSTM

# Load the model
vocab_size = len(word_vocab)
output_size = len(gloss_vocab)
embedding_dim = 128
hidden_dim = 256
n_layers = 2
model = BiDirectLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load('bi_direct_lstm2.pth'))

# Evaluate the model
model.eval()
y_true = []
y_pred = []

for i, (inputs, labels) in tqdm(enumerate(test_loader)):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 2)
    y_true.extend(labels.view(-1).tolist())
    y_pred.extend(preds.view(-1).tolist())
    
# Calculate the F1 and accuracy scores
f1 = f1_score(y_true, y_pred, average='micro')
acc = accuracy_score(y_true, y_pred)
print(f"F1 score: {f1}")
print(f"Accuracy score: {acc}")

# Save the predictions
with open('bi_direct_lstm2_predictions.json', 'w') as f:
    json.dump({'y_true': y_true, 'y_pred': y_pred}, f)
    