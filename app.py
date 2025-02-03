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

# Calculate percentage of Hindi and Marathi words
def calculate_percentages(labeled_words):
    total_words = len(labeled_words)
    hindi_count = sum(1 for _, label in labeled_words if label == 'Hindi')
    marathi_count = total_words - hindi_count
    hindi_percentage = (hindi_count / total_words) * 100 if total_words > 0 else 0
    marathi_percentage = (marathi_count / total_words) * 100 if total_words > 0 else 0
    return hindi_percentage, marathi_percentage

# Streamlit UI
st.title("Hindi-Marathi Language Classifier")
st.write("Identify each word in a sentence as either **Hindi** or **Marathi**.")
st.write("Use language or words which contains hindi and marathi words only preferred in ramanized form, for e.g kai zala, mera naam daksh aahe.")

# Input Text Box
sentence = st.text_input("Enter a sentence:")

# Display Results
if sentence:
    st.subheader("Word-wise Classification:")
    labeled_sentence = label_sentence(sentence)
    for word, label in labeled_sentence:
        st.write(f"**{word}** : {label}")

    st.subheader("Language Percentages:")
    hindi_percentage, marathi_percentage = calculate_percentages(labeled_sentence)
    st.write(f"**Hindi:** {hindi_percentage:.2f}%")
    st.write(f"**Marathi:** {marathi_percentage:.2f}%")
