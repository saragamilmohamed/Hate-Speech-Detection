import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from clean_text import CleanTextTransformer

# Load model and tokenizer
model = tf.keras.models.load_model("model lstm bi.h5")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # same as used in training

# Preprocessing + Prediction
def classify_text(text):
    cleaner = CleanTextTransformer(text_column='tweet', return_tokens=True, join_tokens=True)
    cleaned = cleaner.transform([text])
    sequence = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Hate Speech" if prob >= 0.5 else "Not Hate Speech"
    return label, float(prob), cleaned[0]

# App Config
st.set_page_config(page_title="Hate Speech Classifier", page_icon="🧠", layout="centered")

# Header
st.title("🧠 Hate Speech Detector")
st.markdown("Enter a sentence to detect whether it contains hate speech using a **Biderictional LSTM model**.")

# Input Area
with st.form("hate_speech_form"):
    user_input = st.text_area("📥 Enter your text here:", height=150)
    submitted = st.form_submit_button("🔍 Classify")

    if submitted:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text!")
        else:
            label, prob, cleaned = classify_text(user_input)

            # Result Display
            st.markdown("---")
            st.subheader("🔎 Classification Result")
            if label == "Hate Speech":
                st.markdown(f"<span style='color:red;font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:green;font-weight:bold;'>{label}</span>", unsafe_allow_html=True)

            st.progress(prob if label == "Hate Speech" else 1 - prob, text=f"Confidence: {prob:.2%}")
            st.code(cleaned, language="text")
            st.caption("🧼 Cleaned version of your input")

# Footer (Optional)
st.markdown("""---  
Made with by [Sara Gamil]  
""")
