from util import *
import streamlit as st


@st.cache_resource
def load_rnn_model():
    # Load the pre-trained model with ReLU activation
    return load_model('simple_rnn_imdb.keras')


# Initialize streamlit app
page_title = "Movie Review Sentiment Analysis "
page_icon = "🎬"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")

st.logo('sent_analysis.jpg', icon_image='sent_analysis.jpg')
# Configure "About" Sidebar
configure_about_sidebar()

st.title(page_title)
st.write(':blue[***Unleash the True Emotion Behind Reviews 🎬💬❤️***]')
st.write('This application analyzes movie reviews 🎥📝 to determine whether the sentiment is '
         'positive 😊 or negative 😠, helping you understand the true feelings 💭 behind every critique.')

st.subheader('Movie Review:')
user_input = st.text_area('Movie Review', label_visibility='collapsed', placeholder='Enter a Movie Review')
analyze_button = st.button('Analyze', type='primary', disabled=not user_input)

if analyze_button:

    # Preprocess the user input -- Lower case + encoding + padding
    preprocess_input = preprocess_text(user_input)

    # Load the RNN model
    model = load_rnn_model()

    # Predict sentiment of movie review
    sentiment, prediction = predict_sentiment(preprocess_input, model)

    st.subheader('Sentiment Analysis:')

    # Consider positive sentiment if prediction score is more than 0.5
    if prediction > 0.5:
        st.write(f'Sentiment: :green[***{sentiment}***]')
        st.write(f'Prediction Score: :green[{prediction:.2f}]')
    else:
        st.write(f'Sentiment: :red[***{sentiment}***]')
        st.write(f'Prediction Score: :red[{prediction:.2f}]')


