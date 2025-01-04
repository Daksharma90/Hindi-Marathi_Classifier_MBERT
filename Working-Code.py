from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load the fine-tuned model and tokenizer
model_name = "GautamDaksh/Hindi-Marathi_Classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict language for a single word
def predict_language(word):
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return 'H' if predicted_class == 0 else 'M'

# Process a sentence and label each word
def label_sentence(sentence):
    words = sentence.split()  # Split sentence into words
    labeled_words = [(word, predict_language(word)) for word in words]
    return labeled_words

# Example input
sentence = "Mera naam Daksh aahe"
labeled_sentence = label_sentence(sentence)

# Print the results
for word, label in labeled_sentence:
    print(f"{word}: {label}")
