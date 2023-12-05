from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset


class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

def show_samples(dataset, num_samples=3, seed=42):
    """
    Show num_samples random examples
    """
    sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Article: {example['Text']}'")
        print(f"'>> Summary: {example['Summary']}'")

def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts, val_labels,
                 test_texts, test_labels):
    """
    Prepare input data for model fine-tuning
    """
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    prepare_val = False if val_texts is None or val_labels is None else True
    prepare_test = False if test_texts is None or test_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
    test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

    return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset, freeze_encoder=True,
                        output_dir='./results'):
    """
    Prepare configurations and base model for fine-tuning
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if val_dataset is not None:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
            save_steps=5,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            eval_steps=5,  # number of update steps before evaluation
            warmup_steps=5,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=5
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            tokenizer=tokenizer
        )

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            save_steps=5,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            warmup_steps=5,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=5,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            tokenizer=tokenizer
        )

    return trainer


if __name__ == '__main__':
    # Use first 1000 docs as training data

    dataset = load_dataset("ninadn/indian-legal")
    show_samples(dataset)
    dataset = dataset.filter(lambda x: x["Summary"] is not None) #Remove rows which have no summary
    train_texts, train_labels = dataset['train']['Text'][:1000], dataset['train']['Summary'][:1000]
    val_texts, val_labels = dataset['train']['Text'][1000:1500], dataset['train']['Summary'][1000:1500]
    test_texts, test_labels = dataset['test']['Text'], dataset['test']['Summary']

    # use Pegasus model as base for fine-tuning
    model_name = 'nsi319/legal-pegasus'
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset)
    trainer.train()

    trainer.save_model('pegasus_model')
