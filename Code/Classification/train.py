import numpy as np
import evaluate
import torch
from huggingface_hub import notebook_login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer

from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

data = load_dataset("lighteval/legal_summarization", "BillSum")

print(data.items())

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast = True)

def tokenize_function(examples):
    return tokenizer(examples['article'], padding="max_length", truncation=True)

tokenized_datasets = data.map(tokenize_function, batched=True)

def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Article: {example['article']}'")
        print(f"'>> Summary: {example['summary']}'")

show_samples(data)


