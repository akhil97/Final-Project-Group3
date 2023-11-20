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
        words = cleaned_text.split()[:300]
        st.write("Sample Case Text:")
        st.write(' '.join(words))
    st.subheader("Step 2: Choose a model from the left sidebar")
    model_name = utils.sidebar()

    if model_name:
        st.write(f"Selected model: {model_name}")

    # Perform classification based on the selected model
    if model_name == "Sentiment Analysis":
        sentiment_score = utils.analyze_sentiment(df_file)

        st.write(f"Sentiment Score: {sentiment_score}")




    elif model_name == "LDA Model":

        topic_words = utils.perform_lda(df_file)

        st.subheader("LDA Model Result:")

        for topic_id, top_words in topic_words:
            st.write(f"Topic {topic_id + 1}: {', '.join(top_words)}")

        # Display bar chart for topic distribution

        st.bar_chart({f"Topic {topic_id + 1}": len(top_words) for topic_id, top_words in topic_words})
if __name__ == "__main__":
    main()