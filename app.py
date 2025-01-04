import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "GautamDaksh/Hindi-Marathi_Classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function to predict language for a single word
def predict_language(word):
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return 'Hindi' if predicted_class == 0 else 'Marathi'

# Process a sentence and label each word
def label_sentence(sentence):
    words = sentence.split()
    labeled_words = [(word, predict_language(word)) for word in words]
    return labeled_words

# Streamlit UI
st.title("Hindi-Marathi Language Classifier")
st.write("Identify each word in a sentence as either **Hindi** or **Marathi**.")

# Input Text Box
sentence = st.text_input("Enter a sentence:")

# Display Results
if sentence:
    st.subheader("Word-wise Classification:")
    labeled_sentence = label_sentence(sentence)
    for word, label in labeled_sentence:
        st.write(f"**{word}** : {label}")
