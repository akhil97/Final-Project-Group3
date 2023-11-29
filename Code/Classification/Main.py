import streamlit as st
import utils
import random

from gensim.parsing.preprocessing import preprocess_string

def main():
    st.header("Understand the Topic of Different Legal Text Clauses")
    st.divider()

    st.subheader("Step 1: Choose a sample text to analyze: \n Sample txt")
    df_file = utils.upload_file("Upload text file")

    if df_file is not None:
        # Remove URLs from the text
        cleaned_text = utils.remove_urls(df_file)

        # Display the processed text
        words = cleaned_text.split()[:200]
        st.write("Sample Case Text:")
        st.write(' '.join(words))
    st.subheader("Step 2: Choose a model from the left sidebar")

    model_name = utils.sidebar()

    seed = 42  # You can choose any integer as the seed
    random.seed(seed)
    if model_name == "InLegalBERT":
        predicted_label = utils.load_and_predict_legal_judgment(cleaned_text)
        st.write(f"InLegalBERT Predicted Label: {predicted_label}")



    elif model_name == "RoBERTa":
        sentiment_label, confidence_percentage = utils.analyze_sentiment_roberta(cleaned_text,seed=seed)
        st.write(f"RoBERTa Sentiment Label: {sentiment_label}")
        st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")

    elif model_name == "Hugging Face Transformers":
        sentiment_label, confidence_percentage = utils.analyze_sentiment_transformers(cleaned_text,seed=seed)
        st.write(f"Transformers Sentiment Label: {sentiment_label}")
        st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")

    elif model_name == "sentiment":
        sentiment_label, confidence_percentage = utils.predict_labels_legal_document(cleaned_text)
        st.write(f"Sentiment Label: {sentiment_label}")
        st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")


if __name__ == "__main__":
    main()