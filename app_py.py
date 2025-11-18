import streamlit as st
import joblib
import numpy as np

# Load models
model1 = joblib.load("r_model.pkl")   # Regression model
model2 = joblib.load("cl_model.pkl")  # Classification model

st.title("ML Model Deployment Demo")

st.subheader("Enter Input Values")

# User inputs
x1 = st.number_input("age")
x2 = st.number_input("sex")
x3 = st.number_input("bmi")
x4 = st.number_input("children")
x5 = st.number_input("smoker")
x6 = st.number_input("region")

# Predict button
if st.button("Predict Regression Output"):
    inputs = np.array([[x1, x2, x3, x4, x5, x6]])
    pred = model1.predict(inputs)[0]
    st.success(f"Regression Model Prediction: {pred}")

if st.button("Predict Classification Output"):
    inputs = np.array([[x1, x2, x3, x4, x5, x6]])
    pred = model2.predict(inputs)[0]
    st.info(f"Classification Model Prediction: {pred}")
