import re
import streamlit as st
import spacy
import torch
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Use the sub method to replace all matches with an empty string
    cleaned_text = re.sub(url_pattern, '', text)

    # Additional preprocessing steps
    processed_text = cleaned_text.lower()  # Convert to lowercase, for example

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
            ["Sentiment Analysis","BERT","RoBERTa","Hugging Face Transformers"],
            captions=["Sentiment Analysis", "BERT","RoBERTa","Hugging Face Transformers"],
            index=None,
        )
        return genre


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']



    # Classify sentiment as positive, negative, or neutral
    sentiment_label = "Positive" if sentiment_score >= 0.05 else "Negative" if sentiment_score <= -0.05 else "Neutral"
    st.write(f"Sentiment: {sentiment_label}")

    return sentiment_score


def analyze_sentiment_bert(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    # Get predicted sentiment class (positive, neutral, negative)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to sentiment label
    sentiment_classes = ['negative', 'positive']
    predicted_sentiment = sentiment_classes[predicted_class]

    return predicted_sentiment

def analyze_sentiment_roberta(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    sentiment_score = probabilities[0, 1].item() * 100
    return sentiment_score
def analyze_sentiment_transformers(text, model_name="bert-base-uncased"):
    sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)
    max_length = 512
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    sentiment_scores = []

    for chunk in chunks:
        result = sentiment_analyzer(chunk)
        sentiment_scores.append(result[0]['score'])

    aggregated_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
    return aggregated_sentiment_score

