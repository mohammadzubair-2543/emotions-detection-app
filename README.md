# emotions-detection-app
# 🎭 Emotion Detection App

A Machine Learning-based web application that detects human emotions from text input using NLP techniques and classification models.

---

## 🚀 Live Demo

👉 https://emotions-detection-app-i5flkjas7x4exnor6ecdpf.streamlit.app/
---

## 📌 Features

* Detects emotions from user input text
* Supports emotions like:

  * Joy 😊
  * Fear 😨
  * Anger 😡
  * Love ❤️
  * Sadness 😢
  * Surprise 😲
* Clean and simple UI using Streamlit
* Real-time prediction

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* NLTK (text preprocessing)
* Streamlit (deployment)

---

## ⚙️ How It Works

1. User enters text
2. Text is cleaned (remove symbols, stopwords, stemming)
3. Converted into numerical form using TF-IDF
4. Logistic Regression model predicts the emotion
5. Output is displayed on the screen

---

## 📂 Project Structure

```
EMOTIONDETECTOR/
│
├── app/
│   └── app.py
│
├── models/
│   ├── logistic_regresion.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│
├── data/
│   └── train.txt
│
├── notebook/
│   └── emotion_model.ipynb
│
├── requirements.txt
└── README.md
```



---

## 📊 Model Details

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF
* Dataset: Custom labeled emotion dataset
* Evaluation: Accuracy & classification report

---

## 📌 Future Improvements

* Add Deep Learning model (LSTM)
* Show prediction confidence score
* Improve UI/UX
* Add more emotions

---

