import re
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel,AutoModelForSequenceClassification
import torch
import spacy

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc,accuracy_score,classification_report
import numpy as np
import progressbar
import progressbar
from transformers import BertForSequenceClassification,AutoModelForPreTraining, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
#__________________________________________
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    words_to_remove = ['Indian Kanoon']
    combined_pattern = re.compile('|'.join([url_pattern.pattern] + [re.escape(word) for word in words_to_remove]),
                                  flags=re.IGNORECASE)
    # Use the sub method to replace all matches with an empty string
    cleaned_text = re.sub(combined_pattern, '', text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]', '',
                          cleaned_text)  # removing everything other than these a-zA-Z0-9.,)\-(/?\t
    cleaned_text = re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', cleaned_text)
    cleaned_text = re.sub("\t+", " ", cleaned_text)  # converting multiple tabs and spaces ito a single tab or space
    cleaned_text = re.sub(" +", " ", cleaned_text)
    cleaned_text = re.sub("\.\.+", "", cleaned_text)  # these were the commmon noises in out data, depends on data
    cleaned_text = re.sub("\A ?", "", cleaned_text)

    # dividing into para wrt to roman points
    cleaned_text = re.sub(r"[()[\]\"$']", " ", cleaned_text)  # removing ()[\]\"$' these characters
    cleaned_text = re.sub(r" no.", " number",
                          cleaned_text)  # converting no., nos., co., ltd.  to number, numbers, company and limited
    cleaned_text = re.sub(r" nos.", " numbers", cleaned_text)
    cleaned_text = re.sub(r" co.", " company", cleaned_text)
    processed_text = re.sub(r" ltd.", " limited", cleaned_text)

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
            ["InLegalBERT","CaseInLegalBERT","CustomInLegalBERT"],
            index=None
        )
        return genre


#___________________________________________________________

def inlegal_bert_judgment(text):
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model = AutoModelForSequenceClassification.from_pretrained("law-ai/InLegalBERT")

    inputs = tokenizer(text[-512:], return_tensors="pt")

    with torch.no_grad():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to obtain class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # Get the predicted class label
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return probabilities,predicted_class

def caselaw_bert_judgment(text):

    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InCaseLawBERT")
    model = AutoModelForSequenceClassification.from_pretrained("law-ai/InCaseLawBERT")

    inputs = tokenizer(text[-512:], return_tensors="pt")

    with torch.no_grad():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to obtain class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # Get the predicted class label
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return probabilities, predicted_class

def  custom_bert_judgment(text):
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("law-ai/CustomInLawBERT")
    model = AutoModelForSequenceClassification.from_pretrained("law-ai/CustomInLawBERT")

    inputs = tokenizer(text[-512:], return_tensors="pt")

    with torch.no_grad():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to obtain class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # Get the predicted class label
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return probabilities, predicted_class
# ------------------
import spacy



# Load the spaCy model
nlp = spacy.load("en_legal_ner_trf")

# Function to process text from a file
def process_text_from_file(text):
    # Read text from the file


    doc = nlp(text)

    # Create a dictionary to store entities by their names
    entity_dict = {}

    # Extract and store entities by their names
    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = set()
        entity_dict[ent.label_].add(ent.text)

    return entity_dict