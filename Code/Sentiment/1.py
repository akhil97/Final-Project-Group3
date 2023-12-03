import spacy

# Load the spaCy model
nlp = spacy.load("en_legal_ner_trf")

# Function to process text from a file
def process_text_from_file(file_path):
    # Read text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    doc = nlp(text)

    # Create a dictionary to store entities by their names
    entity_dict = {}

    # Extract and store entities by their names
    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = set()
        entity_dict[ent.label_].add(ent.text)

    # Print entities with the same name together, separated by commas
    for label, entities in entity_dict.items():
        print(f"{label}: {', '.join(entities)}")


# Specify the path to your file
file_path = '/home/ubuntu/NLP_Project/Code/Classification/Sentiment/a.txt'

# Call the function to process text from the file
process_text_from_file(file_path)
