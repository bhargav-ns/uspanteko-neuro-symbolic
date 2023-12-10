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

from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab
from load_data import indices_to_tokens
from model import BiDirectLSTM

parser = argparse.ArgumentParser(description="Load model")

# Add the arguments
parser.add_argument('--student_eval', action='store_true', help='evaluate the student model')
parser.add_argument('--teacher_eval', action='store_true', help='evaluate the teacher model')

# Parse the arguments
args = parser.parse_args()

# Load the model

if args.student_eval:
    model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), 128, 256, 2)
    model.load_state_dict(torch.load('student_model.pth', map_location=torch.device('cpu')))
    
elif args.teacher_eval:
    model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), 128, 512, 2)
    model.load_state_dict(torch.load('teacher_model.pth', map_location=torch.device('cpu')))

# Evaluate the model
model.eval()
y_true = []
y_pred = []

pad_index = gloss_vocab['<pad>']

for i, (inputs, labels) in tqdm(enumerate(test_loader)):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 2)
    labels = labels.view(-1).tolist()
    preds = preds.view(-1).tolist()
    
    y_true.extend([label for label in labels if label != pad_index])
    y_pred.extend([pred for pred, label in zip(preds, labels) if label != pad_index])

inverse_gloss_dict = {v: k for k, v in gloss_vocab.items()}

y_true_labels = [inverse_gloss_dict[idx] for idx in y_true]
y_pred_labels = [inverse_gloss_dict[idx] for idx in y_pred]


print(y_true_labels[0:25])
print(y_pred_labels[0:25])



# Calculate the F1 and accuracy scores
f1 = f1_score(y_true, y_pred, average='micro')
acc = accuracy_score(y_true, y_pred)
print(f"F1 score: {f1}")
print(f"Accuracy score: {acc}")

pdb.set_trace()

# Save the predictions
with open('bi_direct_lstm2_predictions_student.json', 'w') as f:
    json.dump({'y_true': y_true_labels, 'y_pred': y_pred_labels}, f)
    