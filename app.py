import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # keep alphanumeric only
    tokens = [i for i in tokens if i.isalnum()]

    # remove stopwords & punctuation
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]

    # stemming
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])  # 👈 important fix

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚨 Spam Detected")
    else:
        st.header("✅ Not Spam")
