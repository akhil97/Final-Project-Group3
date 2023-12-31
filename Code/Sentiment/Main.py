import streamlit as st
import utils
import random
import pandas as pd
import base64
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc,accuracy_score
import numpy as np


#____________________________________Packages_____________________________________________________
prediction_results = {}


def main():

    st.markdown("<h1 style='font-size: 2 em; color: #000000; font-weight: bold;display: inline-block; text-align: right; white-space: nowrap;position: absolute; right: -250px;min-width: 600px;'> Legal Case Judgement  Prediction and Extracting  Legal Named Entities</h1>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Step 1: Choose a sample text to analyze:</h1>", unsafe_allow_html=True)



    df_file = utils.upload_file("Upload text file")

    if df_file is not None:
        # Remove URLs from the text
        cleaned_text = utils.remove_urls(df_file)

        words = cleaned_text.split()[-200:]
        st.write("Sample Legal Case Text:")
        st.write(' '.join(words))

    st.markdown("<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Step 2: Choose a model from the left sidebar:</h1>",unsafe_allow_html=True)

    model_name = utils.sidebar()

    seed = 42  # You can choose any integer as the seed
    random.seed(seed)
    if model_name == "InLegalBERT":
        probabilities,predicted_class = utils.inlegal_bert_judgment(cleaned_text)

        prediction_label = "Accepted" if predicted_class == 1 else "Rejected"

        st.markdown( "<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Prediction of Legal Judgement:</h1>",unsafe_allow_html=True)
        st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.5em;'>Predicted Class:</span> {prediction_label}",unsafe_allow_html=True)

        # Display prediction and confidence chart with custom colors
        st.write("Prediction Confidence Bar Chart:")
        st.bar_chart(probabilities[0].numpy(), use_container_width=True)

        st.markdown("<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Extracting Legal Named Entities:</h1>",unsafe_allow_html=True)
        entities = utils.process_text_from_file(cleaned_text)
        for label, entities_list in entities.items():
            st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.0em;'>{label}:</span> {', '.join(entities_list)}",unsafe_allow_html=True)
    elif model_name == "InlegaltunedBERT":
        probabilities,predicted_class = utils.inlegal_tune_bert_judgment(cleaned_text)

        prediction_label = "Accepted" if predicted_class == 1 else "Rejected"

        st.markdown( "<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Prediction of Legal Judgement:</h1>",unsafe_allow_html=True)
        st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.5em;'>Predicted Class:</span> {prediction_label}",unsafe_allow_html=True)

        # Display prediction and confidence chart with custom colors
        st.write("Prediction Confidence Bar Chart:")
        st.bar_chart(probabilities[0].numpy(), use_container_width=True)

        st.markdown("<h1 style='font-size: 2.0em; color: #000000; font-weight: bold;'>Extracting Legal Named Entities:</h1>",unsafe_allow_html=True)
        entities = utils.process_text_from_file(cleaned_text)
        for label, entities_list in entities.items():
            st.write(f"<span style='font-weight: bold; color: #001f3f;font-size: 1.0em;'>{label}:</span> {', '.join(entities_list)}",unsafe_allow_html=True)


if __name__ == "__main__":

    main()