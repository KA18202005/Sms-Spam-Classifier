import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# Download only what is really needed
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()

    # ✅ SAFE tokenizer (no punkt / punkt_tab needed)
    tokens = wordpunct_tokenize(text)

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

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚨 Spam Detected")
    else:
        st.header("✅ Not Spam")
