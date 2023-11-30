import re


import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel,AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
#__________________________________________
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    words_to_remove = ['Indian Kanoon']
    combined_pattern = re.compile('|'.join([url_pattern.pattern] + [re.escape(word) for word in words_to_remove]),
                                  flags=re.IGNORECASE)
    # Use the sub method to replace all matches with an empty string
    cleaned_text = re.sub(combined_pattern, '', text)
    cleaned_text = re.sub(r'\d+', '',  cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]','', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]','', cleaned_text) # removing everything other than these a-zA-Z0-9.,)\-(/?\t
    cleaned_text = re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', cleaned_text)
    cleaned_text = re.sub("\t+", " ", cleaned_text) # converting multiple tabs and spaces ito a single tab or space
    cleaned_text = re.sub(" +", " ", cleaned_text)
    cleaned_text = re.sub("\.\.+", "",   cleaned_text)  # these were the commmon noises in out data, depends on data
    cleaned_text = re.sub("\A ?", "",  cleaned_text)

      # dividing into para wrt to roman points
    cleaned_text = re.sub(r"[()[\]\"$']", " ", cleaned_text)  # removing ()[\]\"$' these characters
    cleaned_text = re.sub(r" no.", " number", cleaned_text)  # converting no., nos., co., ltd.  to number, numbers, company and limited
    cleaned_text = re.sub(r" nos.", " numbers", cleaned_text)
    cleaned_text = re.sub(r" co.", " company", cleaned_text)
    cleaned_text = re.sub(r" ltd.", " limited", cleaned_text)

    # Tokenize the cleaned text
    words = word_tokenize(cleaned_text)

    # Get a list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a string
    processed_text = ' '.join(filtered_words)

    return processed_text
#___________________________________________________________
def upload_file(file_name):
    # File uploader (for Streamlit)
    file = st.file_uploader(f"Choose the {file_name}", type=["txt"])

    if file is not None:
        # Read the content of the file as bytes
        content_bytes = file.read()

        if content_bytes:
            # Decode the bytes into a string
            content = content_bytes.decode('utf-8')

            return content
        else:
            st.error("File is empty. Please choose a file with content.")
            return None
    else:
        return None
#______________________________________________________________
def sidebar():
    with st.sidebar:
        genre = st.radio(
            "Choose your model",
            ["InLegalBERT","CustomInLegalBERT","CustomInLegalRoBERTa"],
            index=None
        )
        return genre

#___________________________________________________________

def inlegal_bert_judgment(text):
    model_name = 'law-ai/InLegalBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[0].tolist()

def custom_bert_judgment(text):

    model_path = 'path/to/your/fine_tuned_model'
    tokenizer_path = 'path/to/your/tokenizer'

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[0].tolist()


def custom_roberta_judgment(text):
    model_path = 'path/to/your/fine_tuned_model'
    tokenizer_path = 'path/to/your/tokenizer'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[0].tolist()