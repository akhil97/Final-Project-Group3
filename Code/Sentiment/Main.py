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
# Function to get base64 of a binary file
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set PNG as page background with shadow effect
@st.cache_data
def set_png_as_page_bg(png_file,opacity=0.7):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll;
       opacity: {opacity};
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

# Specify the path to your background image
background_image_path = "/Users/brundamariswamy/Downloads/court.jpg"

# Call the function to set the background image with shadow effect
set_png_as_page_bg(background_image_path)


def main():

    st.markdown("<h1 style='font-size: 4.5em; color: #000000; font-weight: bold;'>Predicting Legal Case Judgement</h1>", unsafe_allow_html=True)

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

        if "InLegalBERT" not in prediction_results:

            predicted_class, predicted_class_binary, probabilities, predicted_prob_positive = utils.inlegal_bert_judgment(
                    cleaned_text)
            prediction_results["InLegalBERT"] = {
                    "predicted_class": predicted_class,
                    "predicted_class_binary": predicted_class_binary,
                    "probabilities": probabilities,
                    "predicted_prob_positive": predicted_prob_positive
            }

                # Access the stored prediction results
            stored_results = prediction_results["InLegalBERT"]

            # Interpret the result
            prediction_label = "Accepted" if stored_results["predicted_class_binary"] == 1 else "Rejected"

            # Convert probabilities to a percentage
            confidence_percentage = round(stored_results["predicted_prob_positive"] * 100, 2)
            confidence_values = [stored_results["predicted_prob_positive"],
                                 1 - stored_results["predicted_prob_positive"]]

            # Display prediction and confidence percentage
            st.write(f"Predicted Class: {prediction_label}")

            # Display prediction and confidence chart with custom colors
            st.subheader("Prediction Confidence Bar Chart:")
            confidence_values = [1 - stored_results["predicted_prob_positive"],
                                 stored_results["predicted_prob_positive"]]
            colors = ['#00ff00'] if stored_results["predicted_class_binary"] == 1 else ['#ff0000']
            st.bar_chart({"Prediction Confidence": confidence_values}, height=200, color=colors)

    elif model_name == "CaseInLegalBERT":
        if "CaseInLegalBERT" not in prediction_results:
            predicted_class, predicted_class_binary, probabilities, predicted_prob_positive = utils.caselaw_bert_judgment(
                cleaned_text)
            prediction_results["CaseInLegalBERT"] = {
                "predicted_class": predicted_class,
                "predicted_class_binary": predicted_class_binary,
                "probabilities": probabilities,
                "predicted_prob_positive": predicted_prob_positive
            }

            # Access the stored prediction results
        stored_results = prediction_results["CaseInLegalBERT"]

        # Interpret the result
        prediction_label = "Accepted" if stored_results["predicted_class_binary"] == 1 else "Rejected"

        # Convert probabilities to a percentage
        confidence_percentage = round(stored_results["predicted_prob_positive"] * 100, 2)
        confidence_values = [stored_results["predicted_prob_positive"], 1 - stored_results["predicted_prob_positive"]]

        # Display prediction and confidence percentage
        st.write(f"Predicted Class: {prediction_label}")

        # Display prediction and confidence chart with custom colors
        st.subheader("Prediction Confidence Bar Chart:")
        confidence_values = [1 - stored_results["predicted_prob_positive"], stored_results["predicted_prob_positive"]]
        colors = ['#00ff00'] if stored_results["predicted_class_binary"] == 1 else ['#ff0000']
        st.bar_chart({"Prediction Confidence": confidence_values}, height=200, color=colors)

    elif model_name == "CustomInLegalBERT":

        if "CustomInLegalBERT" not in prediction_results:
            predicted_class, predicted_class_binary, probabilities, predicted_prob_positive = utils.custom_bert_judgment(
                cleaned_text)
            prediction_results["CustomInLegalBERT"] = {
                "predicted_class": predicted_class,
                "predicted_class_binary": predicted_class_binary,
                "probabilities": probabilities,
                "predicted_prob_positive": predicted_prob_positive
            }

            # Access the stored prediction results
        stored_results = prediction_results["CustomInLegalBERT"]

        # Interpret the result
        prediction_label = "Accepted" if stored_results["predicted_class_binary"] == 1 else "Rejected"

        # Convert probabilities to a percentage
        confidence_percentage = round(stored_results["predicted_prob_positive"] * 100, 2)
        confidence_values = [stored_results["predicted_prob_positive"], 1 - stored_results["predicted_prob_positive"]]

        # Display prediction and confidence percentage
        st.write(f"Predicted Class: {prediction_label}")

        # Display prediction and confidence chart with custom colors
        st.subheader("Prediction Confidence Bar Chart:")
        confidence_values = [1 - stored_results["predicted_prob_positive"], stored_results["predicted_prob_positive"]]
        colors = ['#00ff00'] if stored_results["predicted_class_binary"] == 1 else ['#ff0000']
        st.bar_chart({"Prediction Confidence": confidence_values}, height=200, color=colors)


if __name__ == "__main__":

    main()