import re


import streamlit as st
import spacy
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModel

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
            ["LegalBERT","RoBERTa","Hugging Face Transformers"],

            index=None,
        )
        return genre

def load_and_predict_legal_judgment(text, model_name="InLegalBERT"):
    # Load the specified model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(f"law-ai/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"law-ai/{model_name}")

    # Tokenize the input text and make predictions
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()


    # Assign labels based on predicted class
    if predicted_class == 0:
        prediction_label = "Rejected"
    else:
        prediction_label = "Accepted"

    return prediction_label


def analyze_sentiment_transformers(text, model_name="nlpaueb/legal-bert-base-uncased", threshold=0.5,seed=None):
    if seed is not None:
        # Set a random seed for reproducibility
        torch.manual_seed(seed)
    sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)
    max_length = 512
    chunk_size =  max_length
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    sentiment_scores = []

    for chunk in chunks:
        result = sentiment_analyzer(chunk)
        sentiment_scores.append(result[0]['score'])

    # Calculate the average sentiment score as a confidence measure
    aggregated_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) * 100

    # Convert sentiment score to a binary label
    sentiment_label = 'positive' if aggregated_sentiment_score >= threshold else 'negative'

    return sentiment_label, aggregated_sentiment_score


