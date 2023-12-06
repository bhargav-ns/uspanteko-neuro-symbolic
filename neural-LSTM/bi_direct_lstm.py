import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Load the dataset
data = pd.read_csv('../data/uspanteko_data.csv')
print(data.iloc[0:4])

# Tokenization function
def tokenize(text):
    return text.replace("['", "").replace("']", "").split("', '")

# Apply tokenization
data['segmented_text_tokens'] = data['segmented_text'].apply(tokenize)
data['gloss_tokens'] = data['gloss'].apply(tokenize)

# Build vocabularies
def build_vocab(tokens):
    token_freqs = Counter(token for sentence in tokens for token in sentence)
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_freqs.items())} # +1 for padding
    vocab['<pad>'] = 0
    return vocab

word_vocab = build_vocab(data['segmented_text_tokens'])
gloss_vocab = build_vocab(data['gloss_tokens'])

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

data['segmented_text_indices'] = data['segmented_text_tokens'].apply(lambda x: tokens_to_indices(x, word_vocab))
data['gloss_indices'] = data['gloss_tokens'].apply(lambda x: tokens_to_indices(x, gloss_vocab))

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
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.view(*x.shape, -1)
        out = self.fc(lstm_out)
        return out

# Model parameters
vocab_size = len(word_vocab)
output_size = len(gloss_vocab)
embedding_dim = 128
hidden_dim = 256
n_layers = 2

# Instantiate the model
model = BiLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word_vocab['<pad>'])  # Assuming <pad> is your padding token in the vocabulary

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure outputs and labels have the same shape
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            output = model(inputs)
            val_loss += criterion(output, labels).item()

    print(f"Epoch: {epoch+1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")