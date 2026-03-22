import streamlit as st

st.title("Plant Disease Detection AI")

st.write("Upload a plant leaf image")

file = st.file_uploader("Choose image", type=["jpg","png"])

if file:
    st.image(file)
    st.success("Prediction: Healthy (Demo)")
