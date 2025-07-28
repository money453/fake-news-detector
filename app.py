import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it's **Fake** or **Real**.")

news_text = st.text_area("Paste your news content here")

if st.button("Check"):
    if news_text.strip():
        vect_input = vectorizer.transform([news_text])
        prediction = model.predict(vect_input)[0]
        
        if prediction == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("⚠️ This seems to be **Fake News**.")
    else:
        st.warning("Please enter some text to analyze.")