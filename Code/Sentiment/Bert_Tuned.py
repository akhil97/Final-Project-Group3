import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
import textwrap
import progressbar as pb  # Use a different name for the progressbar module
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json
from transformers import AutoTokenizer
import progressbar
from keras.preprocessing.sequence import pad_sequences

import progressbar

# With this import
from tqdm import tqdm

df = pd.read_csv('/Users/brundamariswamy/Downloads/ILDC_multi/ILDC_multi.csv') # path to multi_dataset
train_set = df.query(" split=='train' ")
test_set = df.query(" split=='test' ")
validation_set = df.query(" split=='dev' ")


def tokenize_and_encode(dataf, tokenizer):
    input_ids = []
    lengths = []

    for i in tqdm(range(len(dataf['text']))):  # Use tqdm instead of progressbar
        sen = dataf['text'].iloc[i]
        sen = tokenizer.tokenize(sen)

        # taking the last 510 tokens
        # you can try out multiple combinations of input tokens as we did in the paper
        if len(sen) > 510:
            sen = sen[len(sen) - 510:]

        encoded_sent = tokenizer.encode(sen, add_special_tokens=True)
        input_ids.append(encoded_sent)
        lengths.append(len(encoded_sent))

    input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="post", padding="post")
    return input_ids, lengths
#%%


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_input_ids, train_lengths = tokenize_and_encode(train_set, tokenizer)
validation_input_ids, validation_lengths = tokenize_and_encode(validation_set, tokenizer)


def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks
train_attention_masks = att_masking(train_input_ids)
validation_attention_masks = att_masking(validation_input_ids)

train_labels = train_set['label'].to_numpy().astype('int')
validation_labels = validation_set['label'].to_numpy().astype('int')
train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_attention_masks
validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

batch_size = 6
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)


lr = 2e-5
max_grad_norm = 1.0
epochs = 3
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 34
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

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

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print("")
print("Training complete!")


# Save the model
output_dir = '/Users/brundamariswamy/Desktop/Sentiment/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
