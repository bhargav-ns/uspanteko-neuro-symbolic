import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.TeacherTagger import TeacherTagger
from models.StudentTagger import StudentTagger
from data.dataset import train_data, word_to_ix, tag_to_ix, ix_to_tag
from data.dataset import dev_data, word_to_ix_dev, tag_to_ix_dev, ix_to_tag_dev
from rules.fol_rules import apply_logic_rules

from transformers import BertTokenizer

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import utils.utils as utils
import config

from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
teacher_model = TeacherTagger(len(word_to_ix), len(tag_to_ix))
student_model = StudentTagger(len(word_to_ix), len(tag_to_ix))

teacher_model = teacher_model.to(device)

# Loss function and optimizer
loss_function = torch.nn.NLLLoss()
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
student_optimizer = optim.Adam(student_model.parameters(), lr=0.01)

# Check if CUDA is available and set device to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

###########
# Training
###########

# Training loop for the teacher model
def train_teacher(save = True):
    losses = []
    teacher_model.train()
    for epoch in tqdm(range(10)):
        epoch_loss = 0
        all_targets = []
        all_predictions = []
        for sentence, tags in tqdm(train_data):
            sentence_in = torch.tensor([word_to_ix[word] for word in sentence.split()], dtype=torch.long).to(device)
            targets = torch.tensor([tag_to_ix[tag] for tag in tags], dtype=torch.long).to(device)

            teacher_model.zero_grad()
            tag_scores = teacher_model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            teacher_optimizer.step()
            
            epoch_loss += loss.item()
            all_targets.extend(targets.tolist())
            all_predictions.extend(tag_scores.argmax(dim=1).tolist())
            
        epoch_loss /= len(train_data)
        losses.append(epoch_loss)
        accuracy = accuracy_score(all_targets, all_predictions)
        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss}, Accuracy: {accuracy}')
        
    # Save the teacher model
    if save:
        utils.save_model(teacher_model, config.TEACHER_MODEL_PATH)


# Training loop for the student model
def train_student(save = True):
    losses = []
    teacher_model.eval()
    student_model.train()
    for epoch in tqdm(range(10)):
        epoch_loss = 0
        all_targets = []
        all_predictions = []
        for sentence, tags in tqdm(train_data):
            sentence_in = torch.tensor([word_to_ix[word] for word in sentence.split()], dtype=torch.long).to(device)
            targets = torch.tensor([tag_to_ix[tag] for tag in tags], dtype=torch.long).to(device)

            # Get teacher predictions and apply logic rules
            with torch.no_grad():
                teacher_output_raw = teacher_model(sentence_in)
                teacher_output = apply_logic_rules(teacher_output_raw, sentence.split())
            
            # Get student predictions
            student_output = student_model(sentence_in)
            
            # Compute NLL loss
            nll_loss = loss_function(student_output, targets)

            # Compute KL divergence
            # kl_loss = F.kl_div(student_output, teacher_output, reduction='batchmean')

            # Compute loss and update student model
            mse_loss = F.mse_loss(student_output, teacher_output)
            loss = 0.2*nll_loss + 0.8*mse_loss
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            
            epoch_loss += loss.item()
            all_targets.extend(targets.tolist())
            all_predictions.extend(student_output.argmax(dim=1).tolist())
        
        epoch_loss /= len(train_data)
        losses.append(epoch_loss)
        accuracy = accuracy_score(all_targets, all_predictions)
        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss}, Accuracy: {accuracy}')
    
    # Save the student model
    if save:
        utils.save_model(student_model, config.STUDENT_MODEL_PATH)
    
if __name__ == "__main__":
    train_teacher()
    train_student()