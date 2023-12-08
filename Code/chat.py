import streamlit as st
from transformers import pipeline
import re
from transformers import AutoTokenizer

def chat():
    st.title("Legal Chat Assistant")
    model_options = {
        "legalbert-large-1.7M-2": "nlpaueb/legal-bert-base-uncased",
        "Legal-Model-B": "model_b_identifier",  
        "Legal-Model-C": "model_c_identifier"   
    }
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), key="legal_assist_model_select")

    model_identifier = model_options[selected_model]
    text_gen_pipeline = pipeline("text-generation", max_length=1000, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    prompt = st.chat_input("How can I assist you?", key="unique_chat_input")
    if prompt:
        cleaned_prompt = clean_input(prompt)

        inputs = tokenizer(cleaned_prompt, return_tensors='pt', max_length=512, truncation=True)
        input_ids = inputs['input_ids']

        with st.spinner('Getting response...'):
            try:
                text_for_pipeline = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                full_response = text_gen_pipeline(text_for_pipeline)[0]["generated_text"]
            except RuntimeError as e:
                full_response = "Sorry, there was an error generating a response. Please try a shorter prompt."

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            with st.chat_message(name="assistant"):
                st.markdown(full_response)

def clean_input(input_text):
    input_text = re.sub(r"(.)\1{2,}", r"\1", input_text)  
    input_text = re.sub(r"[^a-zA-Z0-9\s.,!?']", '', input_text) 
    input_text = input_text.strip() 
    max_length = 500  
    if len(input_text) > max_length:
        input_text = input_text[:max_length]
    return input_text

chat()
