import streamlit as st
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the sentiment analysis model and tokenizer
model_name = "mmr44/fine-tuned-hupd-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_analysis = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Load dataset to get patent numbers
dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
    trust_remote_code=True
)

train_set = dataset_dict['train']
# Convert to DataFrame to get patent numbers
train_df = train_set.to_pandas()
patent_numbers = train_df['patent_number'].unique().tolist()

# Create a dropdown menu for patent application numbers
st.title("Patent Application Sentiment Analysis")

application_number = st.selectbox(
    "Select Patent Application Number",
    patent_numbers  # Populate dropdown with patent numbers from the dataset
)

# Show abstract and claims
selected_patent = train_df[train_df['patent_number'] == application_number].iloc[0]
abstract_text = st.text_area("Abstract", selected_patent['abstract'])
claims_text = st.text_area("Claims", selected_patent['claims'])

# Function to truncate text
def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Submit button
if st.button("Submit"):
    # Prepare the text for analysis
    text_to_analyze = f"Abstract: {abstract_text} Claims: {claims_text}"
    
    # Truncate the text if it's too long
    truncated_text = truncate_text(text_to_analyze, tokenizer)

    # Perform sentiment analysis only if the text is non-empty
    if truncated_text.strip():
        inputs = tokenizer(truncated_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        labels = sentiment_analysis.model.config.id2label
        label = labels[probs.argmax().item()]
        score = probs.max().item()
        
        # Display the result
        st.write(f"Sentiment: {label}, Score: {score}")
    else:
        st.write("The text is too short for analysis.")
