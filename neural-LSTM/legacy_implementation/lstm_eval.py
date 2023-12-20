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
import pdb
import argparse

from model import BiDirectLSTM

from utils import n_layers, student_embed_size, student_hidden_size, teacher_embed_size, teacher_hidden_size, rule_penalty, loss_balancer

from load_data import X_test, y_test, gloss_vocab, word_vocab, train_data_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test, y_test = X_test.to(device), y_test.to(device)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32)

parser = argparse.ArgumentParser(description="Load model")


# Add the arguments
parser.add_argument('--student_eval', action='store_true', help='evaluate the student model')
parser.add_argument('--teacher_eval', action='store_true', help='evaluate the teacher model')

# Parse the arguments
args = parser.parse_args()

# Load the model

if args.student_eval:
    model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), student_embed_size, student_hidden_size, n_layers)
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('student_model.pth'))
    else:        
        model.load_state_dict(torch.load('student_model.pth', map_location=torch.device('cpu')))
    eval_mode = 'student'
    
elif args.teacher_eval:
    model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), teacher_embed_size, teacher_hidden_size, n_layers)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('teacher_model.pth'))
    else:        
        model.load_state_dict(torch.load('teacher_model.pth', map_location=torch.device('cpu')))
    eval_mode = 'teacher'


model.to(device)
# Evaluate the model
model.eval()
y_true = []
y_pred = []

pad_index = gloss_vocab['<pad>']
print("Made it till the loop")
for i, (inputs, labels) in tqdm(enumerate(test_loader)):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 2)
    labels = labels.view(-1).tolist()
    preds = preds.view(-1).tolist()

    y_true.extend([label for label in labels if label != pad_index])
    y_pred.extend([pred for pred, label in zip(preds, labels) if label != pad_index])

print("Loaded models!")
inverse_gloss_dict = {v: k for k, v in gloss_vocab.items()}

y_true_labels = [inverse_gloss_dict[idx] for idx in y_true]
y_pred_labels = [inverse_gloss_dict[idx] for idx in y_pred]


# print(y_true_labels[0:25])
# print(y_pred_labels[0:25])


# Initialize a dictionary to hold counts of each unique token in true labels
token_counts = Counter(y_true_labels)

# Initialize a dictionary to hold error counts for each unique token
error_counts = Counter()

# Iterate over both lists simultaneously
for true, pred in zip(y_true_labels, y_pred_labels):
    # If the labels are not the same, increment the error count for the true label
    if true != pred:
        error_counts[true] += 1

# Calculate the proportion of errors for each unique token
error_proportions = {token: error_counts[token] / count for token, count in token_counts.items()}

# print(error_proportions)

with open('log_file.txt', 'a') as f:

    # Calculate the F1 and accuracy scores
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    print(f"F1 score: {f1}")
    print(f"Accuracy score: {acc}")
    print(f"Train data size: {train_data_size}")
    print(f"Penalty : {rule_penalty}")
    print(f"Loss Balancer : {loss_balancer}")
    f.write(f"{eval_mode} - F1 score: {f1} \n Accuracy score: {acc} \n Train data size: {train_data_size} \n Penalty : {rule_penalty} \n Loss Balancer : {loss_balancer}")
    f.write("\n ========================================= \n")
    
    # pdb.set_trace()

# Save the predictions
# with open('bi_direct_lstm2_predictions_student.json', 'w') as f:
#     json.dump({'y_true': y_true_labels, 'y_pred': y_pred_labels}, f)
