# ======================== IMPORTS ========================
import streamlit as st
import numpy as np
import re
import pickle
import nltk
import os
from nltk.stem import PorterStemmer

# ======================== NLTK SETUP ========================
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ======================== PATH SETUP ========================
# Get absolute path of current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to root → models folder
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

# ======================== LOAD MODELS ========================
lg = pickle.load(open(os.path.join(MODEL_PATH, "logistic_regresion.pkl"), "rb"))
tfidf_vectorizer = pickle.load(open(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"), "rb"))
lb = pickle.load(open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb"))

# ======================== TEXT CLEANING ========================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# ======================== PREDICTION ========================
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    # probability
    probs = lg.predict_proba(input_vectorized)
    confidence = np.max(probs)

    return predicted_emotion, confidence

# ======================== STREAMLIT UI ========================
st.title("🧠 Emotion Detection App")
st.write("Detect emotions from text using Machine Learning")

st.write("### Supported Emotions:")
st.write("Joy | Fear | Anger | Love | Sadness | Surprise")

# Input
user_input = st.text_input("Enter your text:")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        emotion, confidence = predict_emotion(user_input)

        st.success(f"Predicted Emotion: **{emotion}**")
        st.info(f"Confidence: **{round(confidence * 100, 2)}%**")



st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}

.footer a {
    color: #4da6ff;
    text-decoration: none;
}
</style>

<div class="footer">
    <p>Designed & Developed by Mohammad Zubair | 
    <a href="https://github.com/mohammadzubair-2543/emotions-detection-app" target="_blank">
    GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)