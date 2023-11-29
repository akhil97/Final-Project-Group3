import streamlit as st
import utils
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
import numpy as np

#____________________________________Packages_____________________________________________________
def main():
    st.header("Understand the Topic of Different Legal Text Clauses")
    st.divider()

    st.subheader("Step 1: Choose a sample text to analyze: \n Sample txt")
    df_file = utils.upload_file("Upload text file")

    if df_file is not None:
        # Remove URLs from the text
        cleaned_text = utils.remove_urls(df_file)

        # Display the processed text
        words = cleaned_text.split()[-512:]
        st.write("Sample Case Text:")
        st.write(' '.join(words))
    st.subheader("Step 2: Choose a model from the left sidebar")

    model_name = utils.sidebar()

    seed = 42  # You can choose any integer as the seed
    random.seed(seed)
    if model_name == "InLegalBERT":
        predicted_label = utils.load_and_predict_legal_judgment(cleaned_text)
        st.write(f"InLegalBERT Predicted Label: {predicted_label}")



    elif model_name == "InCaseLawBERT":
        sentiment_label, confidence_percentage = utils.analyze_sentiment_roberta(cleaned_text,seed=seed)
        st.write(f"RoBERTa Sentiment Label: {sentiment_label}")
        st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")

    elif model_name == "CustomInLegalBERT":
        true_labels = []
        prediction, probabilities = utils.custom_bert_judgment(cleaned_text)

        # Display the results
        st.header("Prediction Result:")
        st.write(f"Predicted Class: {'Accepted' if prediction == 1 else 'Rejected'}")
        st.write(f"Confidence: {probabilities[prediction]:.2%}")
        # Collect true labels
        true_labels.append(st.radio("True Label:", ["Rejected", "Accepted"]))

        # Evaluate confusion matrix if at least one document is uploaded
        if true_labels:
            st.header("Confusion Matrix:")

            # Convert true labels to binary format (0 for Rejected, 1 for Accepted)
            true_labels_binary = [0 if label == "Rejected" else 1 for label in true_labels]

            # Create confusion matrix
            y_true = np.array(true_labels_binary)
            y_pred = np.array([prediction])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Accepted"])
            disp.plot()
            st.pyplot()

            st.header("Precision-Recall Curve:")
            precision, recall, _ = precision_recall_curve(true_labels_binary, [probabilities[0], probabilities[1]])

            # Calculate AUC
            auc_score = auc(recall, precision)

            # Plot Precision-Recall Curve
            plt.plot(recall, precision, label=f'AUC = {auc_score:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='best')
            st.pyplot(plt)

    elif model_name == "CustomInLegalRoBERTa":
        sentiment_label, confidence_percentage = utils.predict_labels_legal_document(cleaned_text)
        st.write(f"Sentiment Label: {sentiment_label}")
        st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")


if __name__ == "__main__":
    main()