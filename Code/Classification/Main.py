import streamlit as st
import utils
import pandas as pd


def main():
    st.header("Understand the Topic of Different Legal Text Clauses")
    st.divider()
    st.subheader("Step 1: Choose a sample text to analyze: \n Sample txt")
    df_file = utils.upload_file("Upload text file")



    if df_file is None:
        st.stop()

    st.divider()

if __name__ == "__main__":
    main()