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

# Load the csv file using pandas

data = pd.read_csv('../data/uspanteko_data.csv')

data['segmented_text'] = data['segmented_text'].apply(ast.literal_eval)
data['gloss'] = data['gloss'].apply(ast.literal_eval)

print("Loaded df with shape: ", data.shape)

# Get the first 5 rows for the 'segmented_text' and 'gloss' columns
print(data[['segmented_text', 'gloss']].head())
print(type(data['segmented_text'][0]))

# Build vocabularies
def build_vocab(tokens):
    token_freqs = Counter(token for sentence in tokens for token in sentence)
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_freqs.items())} # +1 for padding
    vocab['<pad>'] = 0
    return vocab

word_vocab = build_vocab(data['segmented_text'])
gloss_vocab = build_vocab(data['gloss'])

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

data['segmented_text_indices'] = data['segmented_text'].apply(lambda x: tokens_to_indices(x, word_vocab))
data['gloss_indices'] = data['gloss'].apply(lambda x: tokens_to_indices(x, gloss_vocab))

# Padding sequences and creating Tensor datasets
def pad_and_create_tensors(indices):
    sequences = [torch.tensor(sequence) for sequence in indices]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences

segmented_text_padded = pad_and_create_tensors(data['segmented_text_indices'])
gloss_padded = pad_and_create_tensors(data['gloss_indices'])

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(segmented_text_padded, gloss_padded, test_size=0.2, random_state=42)

# Creating Dataloaders
batch_size = 32
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

print("Train data size: ", len(train_data))
print("Val data size: ", len(val_data))
print("Sample train data: ", train_data[0])

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)


###########
# Model
###########

class BiDirectLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out
    
# Model parameters
vocab_size = len(word_vocab)
output_size = len(gloss_vocab)
embedding_dim = 128
hidden_dim = 256
n_layers = 2

# Instantiate the model
model = BiDirectLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


###########
# Training
###########

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10

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