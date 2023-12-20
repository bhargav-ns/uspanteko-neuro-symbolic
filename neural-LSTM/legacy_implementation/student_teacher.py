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
from utils import vocab_size, output_size, n_layers, student_embed_size, student_hidden_size, teacher_embed_size, teacher_hidden_size

from model import BiDirectLSTM
import time

print("CUDA availability: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rule_violation_penalty(output, gloss_vocab):
    rules = {
        gloss_vocab['PERS']: [gloss_vocab['VT'], gloss_vocab['VI'], gloss_vocab['SREL'], gloss_vocab['S']],
        gloss_vocab['TAM']: [gloss_vocab['VT'], gloss_vocab['VI'], gloss_vocab['PERS']],
        gloss_vocab['VT']: [gloss_vocab['PART'], gloss_vocab['SUF'], gloss_vocab['S'], gloss_vocab['PRON']]
    }
    penalty = 0
    for i in range(len(output) - 1):
        current_morpheme = output[i]
        next_morpheme = output[i+1]
        if current_morpheme in rules and next_morpheme not in rules[current_morpheme]:
            penalty += 1  # Increment penalty for each violation
    return penalty

def custom_loss_function(original_loss, output, gloss_vocab, penalty_scale):
    penalty = rule_violation_penalty(output, gloss_vocab)
    return original_loss + penalty_scale * penalty


# Instantiate the models
teacher_model = BiDirectLSTM(vocab_size, output_size, teacher_embed_size, teacher_hidden_size, n_layers)
teacher_model = teacher_model.to(device)

student_model = BiDirectLSTM(vocab_size, output_size, student_embed_size, student_hidden_size, n_layers)
student_model = student_model.to(device)


def train_teacher(rule_penalty):
    ###########
    # Training
    ###########

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 3
    
    start_time = time.time()
    
    # Training loop
    epochs = 20
    print("Training teacher")
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        val_loss = 0.0
        
        teacher_model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            outputs = teacher_model(inputs)
            CE_Loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            
            loss = custom_loss_function(CE_Loss, outputs, gloss_vocab, penalty_scale=rule_penalty)
            
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
        
        # Check if the validation loss has improved
        if round(val_loss,3) < round(best_val_loss,3):
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # If the validation loss hasn't improved for n_epochs_stop epochs, stop training
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break
        
        print(f"Epoch: {epoch+1}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss}")
    
    end_time = time.time()
    print(f"Training took {end_time-start_time} seconds")
    
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    
def train_student(loss_balancer):
    ###########
    # Training
    ###########

    # Loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 3
    
    start_time = time.time()
    # Training loop
    epochs = 30
    print("Training student")
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
            
            MSE_loss = F.mse_loss(outputs, teacher_output)
            CE_Loss = loss_func(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            
            loss = loss_balancer*MSE_loss + (1-loss_balancer)*CE_Loss
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
        
        # Check if the validation loss has improved
        if round(val_loss,3) < round(best_val_loss,3):
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # If the validation loss hasn't improved for n_epochs_stop epochs, stop training
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break
        
        print(f"Epoch: {epoch+1}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss}")
    
    end_time = time.time()
    print(f"Training took {end_time-start_time} seconds")
    torch.save(student_model.state_dict(), 'student_model.pth')

if __name__ == '__main__':
    train_teacher(0.5)
    train_student(0.8)