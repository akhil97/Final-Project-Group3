import streamlit as st
import utils


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
        words = df_file.split()[:300]
        st.write("Sample Case Text:")
        st.write(' '.join(words))
    st.subheader("Step 2: Choose a model from the left sidebar")
    model_name = utils.sidebar()

    if model_name == "BERT":
        sentiment_score_bert = utils.analyze_sentiment_bert(cleaned_text)
        st.write(f"Predicted Sentiment for BERT: {sentiment_score_bert}")

    elif model_name == "RoBERTa":
        sentiment_score_roberta = utils.analyze_sentiment_roberta(cleaned_text)
        st.write(f"Sentiment Score (RoBERTa): {sentiment_score_roberta}")

    elif model_name == "Hugging Face Transformers":
        sentiment_score_transformers = utils.analyze_sentiment_transformers(cleaned_text)
        st.write(f"Sentiment Score (Hugging Face Transformers): {sentiment_score_transformers}")

if __name__ == "__main__":
    main()