import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model
model = BertForSequenceClassification.from_pretrained('my_feedback_model')
tokenizer = BertTokenizer.from_pretrained('my_feedback_model')
model.eval()

def validate_feedback(text):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**encoding).logits
        pred = torch.argmax(logits, dim=1).item()
    return "Valid" if pred == 1 else "Invalid"

# Streamlit UI
st.title("Zero Education Students Feedback Validation (Transformer-Based)")
st.write("Enter customer feedback below to check if it's valid or invalid:")

user_input = st.text_area("Feedback Text", "")

if st.button("Validate Feedback"):
    if user_input.strip():
        result = validate_feedback(user_input)
        st.success(f"Feedback is **{result}**.")
    else:
        st.warning("Please enter some feedback text.")

st.write("---")
st.write("Example: Try 'Thank you for your quick reply.' or 'This is spam.'")
