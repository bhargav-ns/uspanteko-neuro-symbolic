import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from conllu import parse_incr
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Data Preparation
data_file = open("english-ud-2.0-170801.udpipe.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))

word_sentences = [[token['form'] for token in sentence] for sentence in sentences]
tag_sentences = [[token['upostag'] for token in sentence] for sentence in sentences]

word_sentences_train, word_sentences_val, tag_sentences_train, tag_sentences_val = train_test_split(word_sentences, tag_sentences, test_size=0.2)

# Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels_train = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(word_sentences_train, tag_sentences_train)
]

tokenized_texts_and_labels_val = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(word_sentences_val, tag_sentences_val)
]

tokenized_texts_train = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_train]
labels_train = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_train]

tokenized_texts_val = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_val]
labels_val = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_val]

# Create vocab for tags
unique_tags = set(tag for doc in tag_sentences for tag in doc)
tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}

# Padding input tokens and labels
MAX_LEN = 100
input_ids_train = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_train],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_val = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_val],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tags_train = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_train],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
tags_val = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_val],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")

# Attention masks
attention_masks_train = [[float(i != 0) for i in ii] for ii in input_ids_train]
attention_masks_val = [[float(i != 0) for i in ii] for ii in input_ids_val]

# Convert all inputs and labels into torch tensors
train_inputs = torch.tensor(input_ids_train)
validation_inputs = torch.tensor(input_ids_val)

train_labels = torch.tensor(tags_train)
validation_labels = torch.tensor(tags_val)

train_masks = torch.tensor(attention_masks_train)
validation_masks = torch.tensor(attention_masks_val)

# DataLoader
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Model
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(tag2idx)  # Number of output labels
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Loop
loss_values = []

for epoch_i in range(0, epochs):
    total_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    # Validation
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# Saving the model
# model.save_pretrained('./model_save')
# tokenizer.save_pretrained('./model_save')

# Example of loading the model
# model = BertForTokenClassification.from_pretrained('./model_save')
# tokenizer = BertTokenizer.from_pretrained('./model_save')