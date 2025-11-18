import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load both ML pipelines
reg_model = joblib.load("r_model(1).pkl")
cl_model = joblib.load("cl_model(1).pkl")

st.title("Medical Charges Prediction & Classification App")

# Input fields
age = st.number_input("Age", 0)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 0.0)
children = st.number_input("Children", 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert to DataFrame (required for pipeline)
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Buttons
if st.button("Predict Regression (Charges)"):
    pred = reg_model.predict(input_df)[0]
    st.success(f"Predicted Medical Charges: {pred:.2f}")

if st.button("Predict Classification (Refund / Class)"):
    pred_class = cl_model.predict(input_df)[0]
    st.success(f"Predicted Class: {pred_class}")
