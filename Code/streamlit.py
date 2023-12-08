import streamlit as st
import requests
import tempfile
import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import random
import torch
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import joblib
import chat

REPO_ID = "pile-of-law/legalbert-large-1.7M-2"

files_to_download = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "vocab.txt"
]

# Directory where to save the model
model_dir = "Models/legalbert-large-1.7M-2"

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Download each file
for file in files_to_download:
    file_path = os.path.join(model_dir, file)
    download_path = hf_hub_download(repo_id=REPO_ID, filename=file)
    with open(file_path, 'wb') as f:
        f.write(open(download_path, 'rb').read())


def extract_text_from_document(file):
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


def generate_response_with_selected_model(model, tokenizer, input_tokenized):
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=150,
                                 max_length=250,
                                 early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][
        0]
    return summary


summary = "This is a summary of the document."


def main():
    inject_custom_css()

    st.warning(
        "This application is exclusively created for illustration and demonstration purposes. Please refrain from depending solely on the information furnished by the model.")

    tab1, tab2, tab3 = st.tabs(["Sum It Up!", "Classify This", "Say Hello!"])

    with tab1:
        st.title("Legal Document Summarizer")
        st.write("## Description")
        st.write("This app summarizes legal documents, highlighting key points and clauses.")
        st.write("## Steps")
        st.write("1. Upload a legal document.")
        st.write("2. Choose a summarization model.")
        st.write("3. Wait for the app to process the document.")
        st.write("4. View the summarized key points and clauses.")

        model_choice = st.sidebar.selectbox("Choose a Model", ["Pegasus Legal", "Pegasus Indian Legal"])

        uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.write("Processing...")
            document_text = extract_text_from_document(uploaded_file)

            if document_text:
                if model_choice == "Pegasus Legal":
                    # summary = bert_summarize(document_text)
                    tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
                    model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
                    input_tokenized = tokenizer.encode(document_text, return_tensors='pt', max_length=1024,
                                                       truncation=True)
                    summary = generate_response_with_selected_model(model, tokenizer, input_tokenized)
                elif model_choice == "Pegasus Indian Legal":
                    tokenizer = AutoTokenizer.from_pretrained("akhilm97/pegasus_indian_legal")
                    model = AutoModelForSeq2SeqLM.from_pretrained("akhilm97/pegasus_indian_legal")
                    input_tokenized = tokenizer.encode(document_text, return_tensors='pt', max_length=1024,
                                                       truncation=True)
                    summary = generate_response_with_selected_model(model, tokenizer, input_tokenized)
                else:
                    st.write(
                        "Please choose an appropriate model from one of the following options - BERT, GPT-3, or XLNet.")

                st.write("## Summary")
                st.write("Here's the summarized content of your document:")
                st.write(summary)
            else:
                st.write("Unable to process the document. Please try again with a different file format.")

    with tab2:
        st.title("Legal Document Classifier and Predictor")
        st.write("## Description")
        st.write(
            "This advanced tool is designed to classify legal documents into specific categories and predict outcomes based on their content. It's useful for legal professionals who need quick insights into a document's nature and potential legal outcomes.")
        st.write("## How It Works")
        st.write(
            "1. **Upload a Legal Document:** Begin by uploading a document. Accepted formats include PDF, DOCX, and TXT.")
        st.write("2. **Choose Your Model:** Select from a range of AI models optimized for legal text analysis.")
        st.write(
            "3. **Document Analysis:** The tool will classify the document into categories such as 'Contract', 'Court Judgment', 'Patent', and more.")
        st.write(
            "4. **Outcome Prediction:** Based on the analysis, it will also predict potential outcomes or implications.")

        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.write("Analyzing Document...")
            document_content = extract_text_from_document(uploaded_file)

            if document_content:
                if model_choice == "BERT":
                    # classification_result, prediction = bert_analyze(document_content)
                    classification_result, prediction = "Contract", "Breach of Contract"
                elif model_choice == "RoBERTa":
                    # classification_result, prediction = roberta_analyze(document_content)
                    classification_result, prediction = "Contract", "Breach of Contract"
                elif model_choice == "Legal-BERT":
                    # classification_result, prediction = legal_bert_analyze(document_content)
                    classification_result, prediction = "Contract", "Breach of Contract"
                else:
                    st.write(
                        "Please choose an appropriate model from one of the following options - BERT, RoBERTa, or Legal-BERT.")

                st.write("## Analysis Results")
                st.write("**Document Classification:**", classification_result)
                st.write("**Predicted Outcome:**", prediction)
            else:
                st.error("Unable to process the document. Please try a different format.")

    with tab3:
        st.write("## Connect with Us")
        with st.form("contact_form"):
            st.write("Feel free to reach out to us!")
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")
            submit_button = st.form_submit_button("Submit")

        st.write("### Socials")

        st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css">',
                    unsafe_allow_html=True)

        linkedin_icon = "<i class='fab fa-linkedin'></i>"
        github_icon = "<i class='fab fa-github'></i>"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Akhil Bharadwaj")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/akhil-bharadwaj-mab97/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/akhil97)", unsafe_allow_html=True)
        with col2:
            st.markdown("#### Brunda Mariswamy")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/brunda-mariswamy/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/bmariswamy5/)", unsafe_allow_html=True)
        with col3:
            st.markdown("#### Chirag Lakhanpal")
            st.markdown(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/chiraglakhanpal/)",
                        unsafe_allow_html=True)
            st.markdown(f"{github_icon} [GitHub](https://github.com/ChiragLakhanpal)", unsafe_allow_html=True)


def inject_custom_css():
    custom_css = """
        <style>
            /* General styles */
            html, body {
                font-family: 'Avenir', sans-serif;
            }

            /* Specific styles for titles and headings */
            h1, h2, h3, h4, h5, h6, .title-class  {
                color: #C72C41; 
            }
            a {
                color: #FFFFFF;  
            } 
            /* Styles to make tabs equidistant */
            .stTabs [data-baseweb="tab-list"] {
                display: flex;
                justify-content: space-around; 
                width: 100%; 
            }

            /* Styles for individual tabs */
            .stTabs [data-baseweb="tab"] {
                flex-grow: 1; 
                display: flex;
                justify-content: center; 
                align-items: center; 
                height: 50px;
                white-space: pre-wrap;
                background-color: #C72C41; 
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
                font-size: 90px; 
            }

            /* Styles for the active tab to make it stand out */
            .stTabs [aria-selected="true"] {
                background-color: #EE4540 !important; 
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }
            /* Styles for the tab hover*/
            .stTabs [data-baseweb="tab"]:hover {
                color: #0E1117 !important; 
                font-weight: bold !important; 
            }

        </style>    
    """
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == "__main__":

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a task", ["Summarize or Classify documents", "Legal Chat"])

    if page == "Summarize or Classify documents":

        main()
    elif page == "Legal Chat":
        chat.chat()
