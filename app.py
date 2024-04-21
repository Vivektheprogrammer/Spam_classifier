import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import scipy.sparse

from sklearn.ensemble import ExtraTreesClassifier

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load ETC model and TF-IDF vectorizer
etc_model = joblib.load('etc_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')


# Define ETC prediction function
def etc_predict(X):
    return etc_model.predict(X)


st.title("Spam Classifier for Email and SMS")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess input text
    transformed_sms = transform_text(input_sms)
    # Vectorize preprocessed text using TF-IDF
    vector_input = tfidf.transform([transformed_sms])
    # Predict using ETC model
    result = etc_predict(vector_input)
    # Display prediction result
    if result == 1:
        st.header("Entered message is a Spam")
    else:
        st.header("Entered message is not a Spam")
