import datasets
import pandas as pd
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_metric
from datasets import load_dataset

dataset = load_dataset("ninadn/indian-legal")

print(dataset.items())

def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Text: {example['Text']}'")
        print(f"'>> Summary: {example['Summary']}'")

show_samples(dataset)

device = 'gpu'
model_ckpt = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, add_prefix_space=True, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

dataset = dataset.filter(lambda x: x["Summary"] is not None)

article_len = [len(x['Text']) for x in dataset['train']]
summary_len = [len(x['Summary']) for x in dataset['train']]

data = pd.DataFrame([article_len, summary_len]).T
data.columns = ['Text Length', 'Summary Length']

data.hist(figsize=(15, 5))


