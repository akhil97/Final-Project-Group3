import torch
import trasformer


from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")

model = AutoModelForSequenceClassification.from_pretrained("path/to/locally/downloaded/model/files")

raw_datasets = load_dataset("path/to/locally/downloaded/dataset/files")
