# from student_teacher import train_student, train_teacher
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
# from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab, train_data_size
# from utils import vocab_size, output_size, n_layers, student_embed_size, student_hidden_size, teacher_embed_size, teacher_hidden_size

import json
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

from model import BiDirectLSTM
import time

###########
# DATA LOAD
############

# train_data_size = 1
# rule_penalty = 0
# loss_balancer = 0
rule_penalty = 0.5
for rule_penalty in [0, 0.5, 1]:
    for train_data_size in [1, 0.7, 0.35]:
        for penalty_rate in [0.5,1,2]:
            for loss_balancer in [1, 0.75, 0.5, 0.25, 0]:
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

                vocab_size = len(word_vocab)
                output_size = len(gloss_vocab)

                X_train, X_val, y_train, y_val = train_test_split(segmented_text_padded, gloss_padded, test_size=0.2, random_state=42)
                if train_data_size < 1:
                    # Splitting the dataset
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

                ##############
                ## MODEL TRAIN ##
                ##############


                print("CUDA availability: ", torch.cuda.is_available())
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                def count_parameters(model):
                    return sum(p.numel() for p in model.parameters() if p.requires_grad)



                def rule_violation_penalty(output, gloss_vocab, penalty_rate = 1):
                    rules = {
                        gloss_vocab['PERS']: [gloss_vocab['VT'], gloss_vocab['VI'], gloss_vocab['SREL'], gloss_vocab['S']],
                        gloss_vocab['TAM']: [gloss_vocab['VT'], gloss_vocab['VI'], gloss_vocab['PERS']],
                        gloss_vocab['VT']: [gloss_vocab['PART'], gloss_vocab['SUF'], gloss_vocab['S'], gloss_vocab['PRON']]
                    }
                    penalty = 0
                    total_violation_count = 0
                    total_token_count = 0

                    _, pred_morphemes = torch.max(output, 2)
                    pred_morphemes = pred_morphemes.view(-1).tolist()
                    inverse_gloss_dict = {v: k for k, v in gloss_vocab.items()}
                    # print([inverse_gloss_dict[idx] for idx in pred_morphemes])
                    for i in range(len(pred_morphemes) - 1):
                        current_morpheme = pred_morphemes[i]
                        next_morpheme = pred_morphemes[i+1]
                        if current_morpheme in rules and next_morpheme not in rules[current_morpheme]:
                            penalty += penalty_rate  # Increment penalty for each violation
                            total_violation_count += 1
                            total_token_count += 1
                    # print(gloss_vocab[current_morpheme], gloss_vocab[next_morpheme])
                    # print(total_violation_count)
                    return penalty

                def custom_loss_function(original_loss, output, gloss_vocab, penalty_scale, penalty_rate):
                    penalty = rule_violation_penalty(output, gloss_vocab, penalty_rate)
                    return original_loss + penalty_scale * penalty


                def train_teacher():
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
                    epochs = 30
                    print("Training teacher")
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

                def train_student(loss_balancer, rule_penalty, penalty_rate):
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

                            loss = custom_loss_function(loss, outputs, gloss_vocab, penalty_scale=rule_penalty, penalty_rate=penalty_rate)
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

                n_layers = 2

                student_embed_size = 64
                student_hidden_size = 128

                teacher_embed_size = 128
                teacher_hidden_size = 256


                ###########
                # EVAL#
                ###########


                def model_eval(mode, rule_penalty, loss_balancer, test_loader, gloss_vocab, word_vocab, train_data_size):

                    print("Evaluating model!")

                    if mode == 't':
                        model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), teacher_embed_size, teacher_hidden_size, n_layers)
                        model.to(device)
                        if torch.cuda.is_available():
                            model.load_state_dict(torch.load('teacher_model.pth'))
                        else:        
                            model.load_state_dict(torch.load('teacher_model.pth', map_location=torch.device('cpu')))

                        eval_mode = 'teacher'

                    else:
                        model = BiDirectLSTM(len(word_vocab), len(gloss_vocab), student_embed_size, student_hidden_size, n_layers)
                        if torch.cuda.is_available():
                            model.load_state_dict(torch.load('student_model.pth'))
                        else:        
                            model.load_state_dict(torch.load('student_model.pth', map_location=torch.device('cpu')))
                        eval_mode = 'student'


                    model.to(device)
                    # Evaluate the model
                    model.eval()
                    y_true = []
                    y_pred = []

                    pad_index = gloss_vocab['<pad>']
                    print("Loaded models!")
                    for i, (inputs, labels) in tqdm(enumerate(test_loader)):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 2)
                        labels = labels.view(-1).tolist()
                        preds = preds.view(-1).tolist()

                        y_true.extend([label for label in labels if label != pad_index])
                        y_pred.extend([pred for pred, label in zip(preds, labels) if label != pad_index])

                    print("Past the loop!")
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

                    with open('log_file_2.txt', 'a') as f:

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

                #############
                # MAIN FUNC
                ##############

                # Instantiate the models
                teacher_model = BiDirectLSTM(vocab_size, output_size, teacher_embed_size, teacher_hidden_size, n_layers)
                teacher_model = teacher_model.to(device)

                student_model = BiDirectLSTM(vocab_size, output_size, student_embed_size, student_hidden_size, n_layers)
                student_model = student_model.to(device)

                train_teacher()
                train_student(loss_balancer, rule_penalty, penalty_rate)

    #             print(f"Total violation count = {total_violation_count}")
    #             print(f"Total token count = {total_token_count}")
    #             print(f"Fraction of violations = {total_violation_count/total_token_count}")

                model_eval('t', rule_penalty, loss_balancer, test_loader, gloss_vocab, word_vocab, train_data_size)
                model_eval('s', rule_penalty, loss_balancer, test_loader, gloss_vocab, word_vocab, train_data_size)