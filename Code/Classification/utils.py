import re
import streamlit as st
import pandas as pd


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
def remove_urls(text):
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Use the sub method to replace all matches with an empty string
    cleaned_text = re.sub(url_pattern, '', text)

    return cleaned_text
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
            ["Sentiment Analysis","LDA Model"],
            captions=["Sentiment Analysis", "LDA"],
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

def perform_lda(text):
    processed_text = preprocess_string(text)
    dictionary = corpora.Dictionary([processed_text])
    corpus = [dictionary.doc2bow(processed_text)]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)

    # Get the words for each topic
    topic_words = []
    for topic_id in range(lda_model.num_topics):
        words = [word for word, _ in lda_model.show_topic(topic_id)]
        topic_words.append((topic_id, words))

    return topic_words