# Write the Streamlit app to a file
with open('app.py', 'w') as f:
    f.write("""
import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis App")

sample_text = "Streamlit is an amazing tool for creating data apps!"
text = st.text_area("Enter text:", sample_text)

model_name = st.selectbox(
    "Select a pretrained model:",
    ["distilbert-base-uncased-finetuned-sst-2-english", "nlptown/bert-base-multilingual-uncased-sentiment"]
)

if st.button("Analyze Sentiment"):
    classifier = pipeline("sentiment-analysis", model=model_name)
    result = classifier(text)
    st.write("Sentiment Analysis Result:")
    st.write(result)
    """)