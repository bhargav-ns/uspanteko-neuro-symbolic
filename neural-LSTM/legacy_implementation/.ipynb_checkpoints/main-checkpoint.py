from student_teacher import train_teacher, train_student
from lstm_eval import model_eval

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
from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab, train_data_size
from utils import vocab_size, output_size, n_layers, student_embed_size, student_hidden_size, teacher_embed_size, teacher_hidden_size

import json
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

from model import BiDirectLSTM
import time

for rule_penalty in [0, 0.5, 1]:
    for loss_balancer in [0,0.2,0.4,0.6,0.8,1]:
        train_teacher(rule_penalty)
        train_student(loss_balancer)
        model_eval(mode = 't', rule_penalty=rule_penalty, loss_balancer=loss_balancer)
        model_eval(mode = 's', rule_penalty=rule_penalty, loss_balancer=loss_balancer)