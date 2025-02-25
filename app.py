import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time
from streamlit_extras.let_it_rain import rain

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load Tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Set max sequence length (same as training)
max_length = 15  

# Function to predict sentiment
def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)[0]

    messages = {
        0: "😞 Sorry to hear that! We're always here to improve. Let us know how we can help! ❤️",
        1: "👍 Seems like you feel neutral! Let us know how we can make your experience better! 😊",
        2: "🎉 Awesome! We love to see you happy. Stay positive! 💖"
    }

    return predicted_class, messages[predicted_class]

# Streamlit UI Configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("🔍 AI-Powered Sentiment Analysis")
st.subheader("📌 Enter a sentence, and let AI analyze your sentiment!")

# User Input
user_input = st.text_area("✍️ Type your text here...", "")

if st.button("🔎 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter text to analyze!")
    else:
        with st.spinner("🤖 Analyzing... Please wait!"):
            time.sleep(2)  # Simulate processing time
            sentiment, message = predict_sentiment(user_input)

        st.subheader("📊 Sentiment Result:")
        if sentiment == 0:
            st.error(f"🔴 Negative Sentiment ({sentiment})")
            rain(emoji="💔", font_size=30, falling_speed=5, animation_length="3s")
        elif sentiment == 1:
            st.warning(f"🟡 Neutral Sentiment ({sentiment})")
            rain(emoji="🤔", font_size=30, falling_speed=5, animation_length="3s")
        else:
            st.success(f"🟢 Positive Sentiment ({sentiment})")
            st.balloons()  # Celebration effect for positive sentiment!

        st.info(f"💡 {message}")

# Footer
st.markdown("---")
st.markdown("🚀 AI Sentiment Analysis App using **Streamlit** & **TensorFlow**")
