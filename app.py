import streamlit as st
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import Model  # Import your trained model class
import tensorflow_hub as hub
import numpy as np

# Load the trained model
MODEL_PATH = "Model/yelp_sentiment_multilingual.ckpt"
model = Model.load_from_checkpoint(MODEL_PATH)
model.eval()

# Load the Universal Sentence Encoder for embedding text
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
encoder = hub.load(MODEL_URL)

# Function to predict sentiment
def predict_sentiment(text):
    if not text.strip():
        return None
    
    # Get text embeddings from the Universal Sentence Encoder
    tf_embeddings = encoder([text])  # This is a TensorFlow tensor
    np_embeddings = np.array(tf_embeddings)  # Convert it to a NumPy array
    torch_embeddings = torch.tensor(np_embeddings, dtype=torch.float32).to(model.device)  # Convert to PyTorch tensor

    # Get model prediction
    with torch.no_grad():
        logits = model(torch_embeddings)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    # Determine sentiment label with emoji
    if probs[0][0] > probs[0][1]:
        label = "negative 👎"
    else:
        label = "positive 👍"
    
    return {"label": label, "score": float(max(probs[0])), "text": text}

# Streamlit UI
st.title("Cross-lingual Sentiment Analysis")
st.write("Enter a text below and get a sentiment prediction!")

# User input
user_input = st.text_area("Enter text:", "")

if st.button("Analyze Sentiment"):
    result = predict_sentiment(user_input)
    if result:
        st.write(f"**Sentiment:** {result['label'].capitalize()}")
        st.write(f"**Confidence Score:** {result['score']:.4f}")
    else:
        st.warning("Please enter valid text.")

