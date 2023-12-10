import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

###########
# Model
###########

class StudentTagger(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim = 128, hidden_dim = 256, n_layers = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.dropout = nn.Dropout(dropout)
    
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores