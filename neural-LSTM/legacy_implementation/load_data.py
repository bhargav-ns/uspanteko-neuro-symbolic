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

# Reducing the training data - if train_data_size < 1
train_data_size = 1

print("CUDA availability: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

def indices_to_tokens(indices, vocab):
    return [vocab[idx] for idx in indices]

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



if train_data_size < 1:
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-train_data_size, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Move tensors to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Creating Dataloaders
batch_size = 32
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

print("Train data size: ", len(train_data))
print("Val data size: ", len(val_data))
print("Sample train data: ", train_data[0])

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

