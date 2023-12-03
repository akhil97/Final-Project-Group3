from transformers import AutoTokenizer,AutoModelForSequenceClassification, AutoModel
import torch

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModelForSequenceClassification.from_pretrained("law-ai/InLegalBERT")

# Read text from a file
file_path = "/home/ubuntu/NLP_Project/Code/Classification/Sentiment/a.txt"  # Replace with the actual path to your file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()
text = text[-512:]
# Tokenize the input text
inputs= tokenizer(text, return_tensors="pt")

with torch.no_grad():
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    outputs = model(**inputs)

logits = outputs.logits
# Apply softmax to obtain class probabilities
probabilities = torch.nn.functional.softmax(logits, dim=1)
# Get the predicted class label
predicted_class = torch.argmax(probabilities, dim=1).item()

print("Raw Logits:", logits)
print("Class Probabilities:", probabilities)
print("Predicted Class Label:", predicted_class)