import streamlit as st
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/spam_model.pkl")

model = load_model()

# Website headingpython -m streamlit run app.py
st.title("SMS Spam Detection System")

st.write(
    "Enter your SMS message below "
    "to check whether it is Spam or Not Spam."
)

# SMS input box
message = st.text_area("Enter your SMS message:")

# Prediction button
if st.button("Check Message"):

    if message.strip():

        result = model.predict([message])[0]

        if result == "spam":
            st.error("SPAM MESSAGE DETECTED!")

        else:
            st.success("NOT SPAM - Legitimate Message")

    else:
        st.warning("Please enter a message.")

st.caption("Developed using Python and Machine Learning")

