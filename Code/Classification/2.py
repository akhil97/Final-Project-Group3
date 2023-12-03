import streamlit as st
import spacy
spacy.cli.download("en_legal_ner_trf")
# Load the en_legal_ner_trf model
nlp = spacy.load("en_legal_ner_trf")

def highlight_entities(text):
    # Process the text with the NER model
    doc = nlp(text)

    # Keep track of unique entities
    unique_entities = set()

    # Replace entities with highlighted HTML
    for ent in doc.ents:
        if ent.text not in unique_entities:
            unique_entities.add(ent.text)
            text = text.replace(ent.text, f'<span style="background-color: {get_entity_color(ent.label_)}">{ent.text}</span>')

    return text

def get_entity_color(label):
    # Define colors for different entity labels
    color_mapping = {
        "COURT": "red",
        "PETITIONER": "blue",
        "RESPONDENT": "green",
        "JUDGE": "purple",
        "LAWYER": "orange",
        "DATE": "brown",
        "ORG": "yellow",
        "GPE": "cyan",
        "STATUTE": "pink",
        "PROVISION": "teal",
        "PRECEDENT": "indigo",
        "CASE_NUMBER": "lime",
        "WITNESS": "maroon",
        "OTHER_PERSON": "grey"
        # Add more entity labels and colors as needed
    }

    return color_mapping.get(label, "white")  # Default color for unknown labels

def main():
    # Upload a document
    st.header("Named Entity Recognition in English from Indian Court Judgement Documents")
    st.subheader("This model extracts COURT, JUDGE, PETITIONER, RESPONDENT, DATE, WITNESS, CASE_NUMBER and 6 more entities from Indian Court Judgement Documents")
    st.divider()

    st.subheader("Step 1: Choose Sample Text")

    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

    if uploaded_file is not None:
        # Read the text from the uploaded document
        text = uploaded_file.read().decode("utf-8")

        # Highlight entities in the text
        highlighted_text = highlight_entities(text)

        # Display the highlighted text
        st.components.v1.html(highlighted_text, height=800, width=1000)

if __name__ == "__main__":
    main()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("law-ai/InCaseLawBERT")

# Read text from a file
file_path = "/home/ubuntu/NLP_Project/Code/Classification/Sentiment/a.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text
encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Load the model with a classification head
model = AutoModelForSequenceClassification.from_pretrained("law-ai/InCaseLawBERT")

# Forward pass through the model
output = model(**encoded_input)

# Access the logits (scores) for each class
logits = output.logits

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=1)

# Predicted class (0 for rejected, 1 for accepted)
predicted_class = torch.argmax(probabilities, dim=1).item()

# Print the prediction
if predicted_class == 0:
    prediction_label = "Rejected"
else:
    prediction_label = "Accepted"

print(f"Predicted Label: {prediction_label}")
