import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
# import requests
# import io
# import h5py


@st.cache_resource
def load_imdb_word_index():
    # Load the IMDB dataset word index
    return imdb.get_word_index()


# Load the pre-trained model with ReLU activation
@st.cache_resource
def load_rnn_model():
    url = 'https://raw.githubusercontent.com/mzeeshanaltaf/sentiment_analysis_rnn_imdb/main/simple_rnn_imdb.keras'
    response = requests.get(url)
    model_file = io.BytesIO(response.content)

    with h5py.File(model_file, 'r') as h5file:
        model = load_model(h5file)

    return model


# Function to preprocess user input
def preprocess_text(text):
    word_index = load_imdb_word_index()
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction function
def predict_sentiment(preprocessed_input, model):
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


def configure_about_sidebar():
    st.sidebar.title('About')
    with st.sidebar.expander('Application'):
        st.markdown(''' IMDB Movie Review Sentiment Analysis. Developed using deep learning algorithm (RNN)''')
    with st.sidebar.expander('Technologies Used'):
        st.markdown(''' 
        * TensorFlow
        * Keras
        * Streamlit''')
    with st.sidebar.expander('Contact'):
        st.markdown(''' Any Queries: Contact [Zeeshan Altaf](mailto:zeeshan.altaf@gmail.com)''')
    with st.sidebar.expander('Source Code'):
        st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/sentiment_analysis_rnn_imdb)''')
