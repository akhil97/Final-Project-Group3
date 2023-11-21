import re
import streamlit as st
import spacy
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Use the sub method to replace all matches with an empty string
    cleaned_text = re.sub(url_pattern, '', text)

    # Tokenize the cleaned text
    words = word_tokenize(cleaned_text)

    # Get a list of English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a string
    processed_text = ' '.join(filtered_words)

    return processed_text
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
def sidebar():
    with st.sidebar:
        genre = st.radio(
            "Choose your model",
            ["BERT","RoBERTa","Hugging Face Transformers"],
            captions=["BERT","RoBERTa","Hugging Face Transformers"],
            index=None,
        )
        return genre



def analyze_sentiment_bert(text):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load legal BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move model and inputs to GPU if available
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():

        outputs = model(**inputs)

    # Get the logits from the output dictionary
    logits = outputs.logits

    # Get predicted sentiment class (positive, neutral, negative)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to sentiment label
    sentiment_classes = ['Negative', 'Positive']
    predicted_sentiment = sentiment_classes[predicted_class]

    return predicted_sentiment

def analyze_sentiment_roberta(text, threshold=0.5):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RoBERTa tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("saibo/legal-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("saibo/legal-roberta-base")

    # Move model and inputs to GPU if available
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Perform inference on the GPU
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits from the output dictionary
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class (0 for negative, 1 for positive)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map predicted class to sentiment label based on a threshold
    sentiment_label = 'Positive' if probabilities[0, 1].item() >= threshold else 'Negative'

    return sentiment_label


def analyze_sentiment_transformers(text, model_name="nlptown/bert-base-multilingual-uncased-sentiment", threshold=0.5):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and pad the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs.to(device)

    # Forward pass to get sentiment logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calculate softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Extract the probability of the positive class
    positive_probability = probabilities[0][1].item()

    # Convert sentiment score to a binary label
    sentiment_label = 'Positive' if positive_probability >= threshold else 'Negative'

    return sentiment_label



