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


def get_feature(batch):
    encodings = tokenizer(batch['Text'], text_target=batch['Summary'],
                          max_length=512, truncation=True, padding='max_length')

    input_encodings = {'input_ids': encodings['input_ids'],
                       'attention_mask': encodings['attention_mask'],
                       'labels': encodings['labels']}
    return input_encodings

data_pt = dataset.map(get_feature, batched = True)

print(data_pt)

columns = ['input_ids', 'labels', 'attention_mask']
data_pt.set_format(type='torch', columns=columns)

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

training_args = TrainingArguments(
    output_dir = 'bart_legal',
    num_train_epochs = 3,
    warmup_steps = 500,
    learning_rate = 1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay = 0.01,
    logging_steps = 10,
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_steps = 1e6,
    fp16 = True,
    gradient_accumulation_steps = 16
)

trainer = Trainer(model = model, args = training_args, tokenizer = tokenizer, data_collator = data_collator, train_dataset = data_pt['train'], eval_dataset = data_pt['test'])

trainer.train()