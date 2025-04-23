# 🧠 Hate Speech Detection Web App

A deep learning-powered web application that detects hate speech in real-time user input using RNN, LSTM, and BiLSTM models. This project aims to raise awareness and help reduce online toxicity by providing a simple, accessible detection tool.

## 🚀 Live Demo
Try the app on **Streamlit**: [Live App Link]([https://your-streamlit-app-link](http://192.168.1.4:8501))

---
## 📂 Project Structure

📁 hate-speech-detection ├── 📄 app.py # Streamlit web application ├── 📁 model/ # Trained models (RNN, LSTM, BiLSTM) ├── 📁 utils/ # Preprocessing and tokenization scripts ├── 📄 requirements.txt # Python dependencies ├── 📄 README.md # Project overview and documentation └── 📄 LICENSE

---

## 💡 Project Highlights

- **Text Preprocessing**: Tokenization, stopword removal, padding.
- **Deep Learning Models**: 
  - RNN
  - LSTM
  - Bidirectional LSTM
- **UI Features**:
  - User-friendly input box
  - Real-time prediction with confidence bar
  - Prediction label: "Hate Speech" or "Not Hate Speech"
- **Deployment**: Built using Streamlit for fast and easy deployment.

## 📊 Dataset

The model was trained on a curated dataset containing labeled tweets and social media content with two classes:
- `0` — Not Hate Speech
- `1` — Hate Speech

Note: The dataset includes examples of bullying, xenophobia, and offensive language.

## ⚙️ Installation & Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/hate-speech-detection.git
cd hate-speech-detection
```
---
## Dependencies
  -  Python
  -  TensorFlow / Keras
  -  Streamlit
  -  NLTK
  -  NumPy / Pandas
