import re
import streamlit as st
import pandas as pd


import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

def upload_file(file_name, num_words=300):
    # File uploader (for Streamlit)
    file = st.file_uploader(f"Choose the {file_name}", type=["txt"])

    if file is not None:
        # Read the content of the file as bytes
        content_bytes = file.read()

        # Decode the bytes into a string
        content = content_bytes.decode('utf-8')

        # Split the content into words
        words = content.split()

        # Display the first 300 words
        st.write(' '.join(words[:num_words]))


