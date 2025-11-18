import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration and Loading ---

# 1. Load the Models and Scaler
try:
    # Load the Regression Model (r_model.pkl - Stacking)
    reg_model = joblib.load('r_model.pkl')
    # Load the Classification Model (cl_model.pkl - SVM)
    cls_model = joblib.load('cl_model.pkl')
    # Load the fitted StandardScaler
    scaler = joblib.load('scaler.pkl')
    
    # Set the page configuration
    st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
    
except FileNotFoundError:
    st.error("""
    ⚠️ **Deployment Error:** Could not find the required files.
    Please ensure the following three files are in the same directory as 'app_py.py':
    1. **`r_model.pkl`** (Regression Model: Stacking)
    2. **`cl_model.pkl`** (Classification Model: SVM)
    3. **`scaler.pkl`** (Fitted StandardScaler)
    """)
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Loading Error:** An unexpected error occurred: {e}")
    st.stop()


# --- Preprocessing Function (The Fix) ---

def preprocess_input(age, sex, bmi, children, smoker, region):
    """Applies the exact preprocessing steps from the training notebook."""
    
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
    # Mapping based on your training notebook: 'female': 1, 'male': 0 and 'yes': 1, 'no': 0
    input_df['sex'] = input_df['sex'].map({'male': 0, 'female': 1})
    input_df['smoker'] = input_df['smoker'].map({'yes': 1, 'no': 0})
    
    # 3. Apply One-Hot Encoding (region)
    input_df = pd.get_dummies(input_df, columns=['region'])
    
    # 4. Define and ensure all 9 required features exist and are in the correct order
    required_cols = [
        'age', 'sex', 'bmi', 'children', 'smoker', 
        'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
    ]

    # Add missing OHE columns (the 3 regions not selected) and set them to 0
    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match the training data feature vector
    final_features = input_df[required_cols].copy() 

    # 5. Apply StandardScaler to numerical features (age, bmi, children)
    numerical_cols = ['age', 'bmi', 'children']
    
    # Use the loaded scaler.transform()
    scaled_values = scaler.transform(final_features[numerical_cols])
    
    # Update the scaled columns in the DataFrame
    final_features.loc[:, numerical_cols] = scaled_values
    
    return final_features


# --- Streamlit UI (Input Collection) ---

st.title('🩺 Medical Cost Prediction App')
st.markdown("Enter the patient's details below to get an estimated charge and risk category.")

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


# --- Prediction Logic ---

if st.button('Predict Medical Charges and Risk'):
    
    # Preprocess the inputs to create the model's feature vector
    try:
        processed_data = preprocess_input(age, sex, bmi, children, smoker, region)
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()
        
    st.subheader('Prediction Results')
    st.balloons()
    
    
    # --- REGRESSION PREDICTION ---
    
    # 1. Make the regression prediction (output is log-transformed)
    log_prediction = reg_model.predict(processed_data)
    
    # 2. Inverse transform (exponentiate) the prediction to get the final dollar amount
    # Your charges column was log-transformed, so we must use np.exp()
    predicted_charge = np.exp(log_prediction)[0]
    
    
    # --- CLASSIFICATION PREDICTION ---
    
    # 1. Make the classification prediction (e.g., 0 or 1)
    predicted_class_encoded = cls_model.predict(processed_data)[0]
    
    # 2. Map the encoded class back to a meaningful category
    if predicted_class_encoded == 1:
        cost_category = "HIGH COST"
        category_color = "red"
    else:
        cost_category = "LOW COST"
        category_color = "green"
        
    # --- DISPLAY RESULTS ---
    
    col_reg, col_cls = st.columns(2)
    
    with col_reg:
        st.metric(label="Estimated Medical Charge (Regression)", 
                  value=f"${predicted_charge:,.2f}")
        st.markdown(f"""
        This value is predicted by the **Stacking Regressor** model.
        """)
        
    with col_cls:
        st.markdown(f"### Predicted Risk Category (Classification)")
        st.markdown(f"# :{category_color}[{cost_category}]")
        st.markdown(f"""
        This category is predicted by the **SVM Classifier** model.
        """)
