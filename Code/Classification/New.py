import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load your CSV file
df = pd.read_csv('/home/ubuntu/NLP_Project/Code/Classification/representative_judgement_sample (1).csv')

# Data Preprocessing
df['Text'] = df['Text'].str.lower().replace('[^a-zA-Z\s]', '', regex=True)

stop_words = set(stopwords.words('english'))
df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Use LabelEncoder to convert string labels to numeric labels
label_encoder = LabelEncoder()
train_df['Numeric_Labels'] = label_encoder.fit_transform(train_df['Case_Type'])
print(train_df['Numeric_Labels'].unique())
val_df['Numeric_Labels'] = label_encoder.transform(val_df['Case_Type'])

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('nlpaueb/legal-bert-base-uncased', num_labels=len(label_encoder.classes_))

# Tokenize and convert to PyTorch tensors
def tokenize_data(texts, labels, max_length=512):
    inputs = tokenizer(texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels.values, dtype=torch.long)
    return inputs, labels

train_inputs, train_labels = tokenize_data(train_df['Text'], train_df['Numeric_Labels'])
val_inputs, val_labels = tokenize_data(val_df['Text'], val_df['Numeric_Labels'])

# Create DataLoader
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
val_predictions = []
val_true_labels = []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = batch['labels'].cpu().numpy()
        val_predictions.extend(predictions)
        val_true_labels.extend(true_labels)

# Calculate accuracy
accuracy = accuracy_score(val_true_labels, val_predictions)
print(f"Validation Accuracy: {accuracy}")
model.save_pretrained("/home/ubuntu/NLP_Project/Code/Classification/fine_tuned_legal_text_model")
tokenizer.save_pretrained("/home/ubuntu/NLP_Project/Code/Classification/fine_tuned_legal_text_tokenize")
