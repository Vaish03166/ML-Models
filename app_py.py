import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration and Loading ---

try:
    # Load the Regression Model (r_model.pkl - Stacking)
    reg_model = joblib.load('r_model.pkl')
    # Load the Classification Model (cl_model.pkl)
    cls_model = joblib.load('cl_model.pkl')
    # Load the fitted StandardScaler
    scaler = joblib.load('scaler.pkl')
    
    st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
    
except FileNotFoundError:
    st.error("""
    ⚠️ **Deployment Error:** Could not find the required files.
    Please ensure **`r_model.pkl`**, **`cl_model.pkl`**, and **`scaler.pkl`** are in the same directory.
    """)
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Loading Error:** An unexpected error occurred: {e}")
    st.stop()


# --- Single Preprocessing Function (6 Features) ---

def preprocess_input(age, sex, bmi, children, smoker, region, scaler):
    """
    Applies the 6-feature Label Encoding and Scaling used for both models.
    """
    
    # 1. Create Base DataFrame
    data = {
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 
        'smoker': [smoker], 'region': [region]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Apply Label Encoding for ALL categorical features
    # Mapping based on LabelEncoder alphabetical output:
    input_df['sex'] = input_df['sex'].map({'female': 0, 'male': 1})
    input_df['smoker'] = input_df['smoker'].map({'no': 0, 'yes': 1})
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    input_df['region'] = input_df['region'].map(region_map)
    
    # 3. Define the 6 required columns in the correct order
    required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'] 
    final_features = input_df[required_cols].copy()
    
    # 4. Apply StandardScaler to numerical features
    numerical_cols = ['age', 'bmi', 'children']
    
    # Scale Numerical Features
    final_features.loc[:, numerical_cols] = scaler.transform(final_features[numerical_cols])

    return final_features


# --- Streamlit UI (Input Collection) ---

st.title('🩺 Medical Cost Prediction App')
st.markdown("Enter the patient's details below to get an estimated charge and risk category.")

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
    
    # 1. Preprocess the inputs to get the 6 features
    try:
        processed_features = preprocess_input(age, sex, bmi, children, smoker, region, scaler)
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()
        
    
    st.subheader('Prediction Results')
    st.balloons()
    
    
    # --- REGRESSION PREDICTION (6 features) ---
    
    log_prediction = reg_model.predict(processed_features)
    
    # FIX: Handle the possibility that the model output is a string/non-numeric due to internal error
    try:
        # Flatten the output array and cast the first element to float
        log_prediction_value = float(log_prediction.flatten()[0])
    except (TypeError, ValueError):
        st.error(f"""
        **Critical Error:** The Regression Model (`r_model.pkl`) returned a value that is not a number. 
        This means the model failed internally. Please re-check the integrity of your `r_model.pkl` file.
        """)
        st.stop()

    # Inverse transform (exponentiate) the guaranteed float value
    predicted_charge = np.exp(log_prediction_value)
    
    
    # --- CLASSIFICATION PREDICTION (6 features) ---
    
    # Make the classification prediction
    predicted_class_encoded = cls_model.predict(processed_features)[0]
    
    # Map the encoded class back to a meaningful category (0=Low, 1=High)
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
        This value is predicted by the **Stacking Regressor** model (6 features).
        """)
        
    with col_cls:
        st.markdown(f"### Predicted Risk Category (Classification)")
        st.markdown(f"# :{category_color}[{cost_category}]")
        st.markdown(f"""
        This category is predicted by the **Classifier** model (6 features).
        """)
