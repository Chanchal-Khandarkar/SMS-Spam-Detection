import streamlit as st
import joblib
import re
import string

# Load saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Same cleaning function used during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'subject:', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("📧 Spam Email Detector")
st.write("Paste an email below to check if it's spam or not.")

user_input = st.text_area("Email text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.error(f"🚨 This looks like SPAM (confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ This looks like a legitimate email (confidence: {probability[0]*100:.1f}%)")