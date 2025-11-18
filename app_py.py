import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration and Loading ---

# 1. Load the Model and Scaler
try:
    # Load the trained XGBoost Regressor model
    model = joblib.load('xgb_regressor.pkl')
    # Load the fitted StandardScaler used on numerical features
    scaler = joblib.load('scaler.pkl')
    
    # Set the page configuration
    st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
    
except FileNotFoundError:
    st.error("⚠️ **Deployment Error:** Could not find 'xgb_regressor.pkl' or 'scaler.pkl'. Please ensure both files are in the same directory as 'app_py.py'.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Loading Error:** An unexpected error occurred: {e}")
    st.stop()


# --- Streamlit UI (Input Collection) ---

st.title('🩺 Medical Cost Prediction App')
st.markdown("""
Enter the patient's details below to get an estimated medical insurance charge.
""")

# Input Columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider('Age', 18, 64, 30)
    sex = st.selectbox('Sex', ['female', 'male'])
    
with col2:
    bmi = st.number_input('BMI (e.g., 25.0)', min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider('Children', 0, 5, 0)
    
with col3:
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])


# --- Preprocessing Function (The Fix for ValueError) ---

def preprocess_input(age, sex, bmi, children, smoker, region):
    
    # 1. Create a DataFrame from the raw inputs
    data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Apply Label/Binary Encoding (sex, smoker)
    # The notebook used 'female': 1, 'male': 0
    input_df['sex'] = input_df['sex'].map({'male': 0, 'female': 1})
    # The notebook used 'yes': 1, 'no': 0
    input_df['smoker'] = input_df['smoker'].map({'yes': 1, 'no': 0})
    
    # 3. Apply One-Hot Encoding (region)
    input_df = pd.get_dummies(input_df, columns=['region'])
    
    # 4. Ensure all 9 required features exist and are in the correct order
    # The required features must match the training set exactly.
    required_cols = [
        'age', 'sex', 'bmi', 'children', 'smoker', 
        'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
    ]

    # Add missing OHE columns (the 3 regions not selected) and set them to 0
    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match the training data feature vector
    final_features = input_df[required_cols]

    # 5. Apply StandardScaler to numerical features (age, bmi, children)
    numerical_cols = ['age', 'bmi', 'children']
    
    # Use the loaded scaler.transform()
    scaled_values = scaler.transform(final_features[numerical_cols])
    
    # Update the scaled columns in the DataFrame
    final_features.loc[:, numerical_cols] = scaled_values
    
    return final_features


# --- Prediction Logic ---

if st.button('Predict Medical Charges'):
    
    # Preprocess the inputs to create the model's feature vector
    try:
        processed_data = preprocess_input(age, sex, bmi, children, smoker, region)
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()
        
    
    # 1. Make the prediction (output is log-transformed)
    log_prediction = model.predict(processed_data)
    
    # 2. Inverse transform (exponentiate) the prediction to get the final dollar amount
    prediction = np.exp(log_prediction)[0]
    
    # 3. Display the result
    st.subheader('Prediction Results')
    st.balloons()
    
    st.markdown(f"""
    Based on the inputs provided, the estimated medical insurance charge is:
    
    # :green[${prediction:,.2f}]
    
    *Note: This is an estimation based on a machine learning model (XGBoost Regressor).*
    """)
