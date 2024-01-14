import transformers
from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from bert_tu_viet import BertCNNForSequenceClassification
from datasets import load_dataset, load_metric, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import os, torch, pandas as pd
import numpy as np
import re
import torch.nn as nn

category_ids = {'negative': 0, 'positive': 1}

model_name_or_path = "bert-base-uncased"
model = BertCNNForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=len(category_ids)
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

df_train = pd.read_csv('data/bert_train.csv').dropna().copy(deep=True).reset_index(drop=True)
df_test = pd.read_csv('data/test.csv').dropna().copy(deep=True).reset_index(drop=True)

max_input_length = 64

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

HASHTAG_SPEC = 'HASHTAG'
USER_SPEC = 'USER'
URL_SPEC = 'URL'
EMOJI_SPEC = 'EMOJI'

urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
tagPattern        = '#[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

def preprocess_text(tweet):
    tweet = tweet.lower()    
        # Replace all URls with 'URL'
    tweet = re.sub(urlPattern, f' {URL_SPEC} ' , tweet)
        # Replace all emojis.
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji, f" {EMOJI_SPEC} " + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
    tweet = re.sub(userPattern, f' {USER_SPEC} ' , tweet)   
    tweet = re.sub(tagPattern, f' {HASHTAG_SPEC} ' , tweet)
    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " " , tweet)
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
    return tweet

def preprocess_data(examples):
    text_inps = [preprocess_text(text) for text in examples['text']]
    model_inputs = tokenizer(text_inps, max_length=max_input_length, truncation=True, pad_to_max_length=True)
    labels = [category_ids[topic] for topic in examples['target']]
    model_inputs["label"] = labels
    return model_inputs

train_datasets = Dataset.from_pandas(df_train).map(preprocess_data, batched=True, num_proc=8)
test_datasets = Dataset.from_pandas(df_test).map(preprocess_data, batched=True, num_proc=8)

batch_size = 100
model_dir = f"bertcnn"
os.makedirs(model_dir, exist_ok=True)

args = TrainingArguments(
    model_dir,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=20,
    fp16=True,
    report_to="tensorboard"
)

data_collator = DataCollatorWithPadding(tokenizer)
round_func = lambda x : round(x * 100, 2)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1)
    with open('logs.txt', 'a') as f:
        f.write(classification_report(labels, predictions))
        f.write('\n\n')
    return {
        'f1': round_func(f1_score(labels, predictions, average='macro')), 
        'accuracy': round_func(accuracy_score(labels, predictions)), 
        'precision': round_func(precision_score(labels, predictions, average='macro')), 
        'recall': round_func(recall_score(labels, predictions, average='macro'))
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=test_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

trainer.train()