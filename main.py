import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()


@st.cache_resource
def load_rnn_model():
    print('Model has been loaded successfully!')
    # Load the pre-trained model with ReLU activation
    return load_model('simple_rnn_imdb.h5')


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction function
def predict_sentiment(preprocessed_input, model):
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


model = load_rnn_model()

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to determine if it is a positive or negative sentiment')

user_input = st.text_area('Movie Review')
classify_button = st.button('Classify', type='primary', disabled=not user_input)

if classify_button:
    preprocess_input = preprocess_text(user_input)
    sentiment, prediction = predict_sentiment(preprocess_input, model)

    if prediction > 0.5:
        st.write(f'Sentiment: :green[***{sentiment}***]')
    else:
        st.write(f'Sentiment: :red[***{sentiment}***]')

    st.write(f'User Input: {user_input}')
    st.write(f'Prediction Score: {prediction:.2f}')
