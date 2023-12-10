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

import torch.nn.functional as F
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

# Instantiate the models
teacher_model = BiDirectLSTM(vocab_size, output_size, 128, 512, n_layers)
teacher_model = teacher_model.to(device)

student_model = BiDirectLSTM(vocab_size, output_size, 128, 256, n_layers)
student_model = student_model.to(device)

def train_teacher():
    ###########
    # Training
    ###########

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

    # Training loop
    epochs = 10

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        val_loss = 0.0
        
        teacher_model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            outputs = teacher_model(inputs)
            loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        teacher_model.eval()
        for i, (inputs, labels) in tqdm(enumerate(val_loader)):
            outputs = teacher_model(inputs)
            loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch: {epoch+1}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss}")

    torch.save(teacher_model.state_dict(), 'bi_direct_lstm2.pth')

def train_student():
    ###########
    # Training
    ###########

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)

    # Training loop
    epochs = 15

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        val_loss = 0.0
        
        student_model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            outputs = student_model(inputs)
            
            # Get teacher predictions and apply logic rules
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            
            loss = F.mse_loss(outputs, teacher_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        student_model.eval()
        for i, (inputs, labels) in tqdm(enumerate(val_loader)):
            outputs = student_model(inputs)
            loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch: {epoch+1}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss}")

    torch.save(student_model.state_dict(), 'bi_direct_lstm2.pth')

if __name__ == '__main__':
    train_teacher()
    train_student()