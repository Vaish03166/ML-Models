import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration and Loading ---

# 1. Load the Models and Scaler
try:
    # Load the Regression Model (r_model.pkl - Stacking)
    reg_model = joblib.load('r_model.pkl')
    # Load the Classification Model (cl_model.pkl - RandomForestClassifier/SVM)
    cls_model = joblib.load('cl_model.pkl')
    # Load the fitted StandardScaler
    scaler = joblib.load('scaler.pkl')
    
    st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
    
except FileNotFoundError:
    st.error("""
    ⚠️ **Deployment Error:** Could not find the required files.
    Please ensure the following three files are in the same directory as 'app_py.py':
    1. **`r_model.pkl`** (Regression Model)
    2. **`cl_model.pkl`** (Classification Model)
    3. **`scaler.pkl`** (Fitted StandardScaler)
    """)
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Loading Error:** An unexpected error occurred: {e}")
    st.stop()


# --- Dual Preprocessing Function (THE FIX for 6 vs 8 features) ---

def preprocess_input(age, sex, bmi, children, smoker, region, scaler):
    """
    Applies dual preprocessing: 8 features for Regression (OHE, drop_first=True), 
    6 features for Classification (Label Encoding).
    """
    
    # 1. Create Base DataFrame
    data = {
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 
        'smoker': [smoker], 'region': [region]
    }
    input_df = pd.DataFrame(data)
    
    
    # --- 2. Prepare Features for CLASSIFICATION (6 features: Label Encoding) ---
    cls_df = input_df.copy()
    
    # Label Encoding based on LabelEncoder alphabetical output (Snippet 6 confirmation)
    cls_df['sex'] = cls_df['sex'].map({'female': 0, 'male': 1})
    cls_df['smoker'] = cls_df['smoker'].map({'no': 0, 'yes': 1})
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    cls_df['region'] = cls_df['region'].map(region_map)
    
    # Define and order the 6 required columns
    cls_required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    cls_features = cls_df[cls_required_cols].copy()
    
    
    # --- 3. Prepare Features for REGRESSION (8 features: OHE, drop_first=True) ---
    reg_df = input_df.copy()
    
    # Apply OHE with drop_first=True (Snippet 4 confirmation)
    reg_df = pd.get_dummies(reg_df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # Define the 8 required columns based on the dropped-first categories
    reg_required_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 
                         'region_northwest', 'region_southeast', 'region_southwest']
    
    # Add missing OHE columns (3 regions not selected and the OHE features) and set them to 0
    for col in reg_required_cols:
        if col not in reg_df.columns:
            reg_df[col] = 0
            
    # Reorder columns to match the training data feature vector
    reg_features = reg_df[reg_required_cols].copy() 

    
    # --- 4. Apply StandardScaler (Scaling numerical features on BOTH feature sets) ---
    numerical_cols = ['age', 'bmi', 'children']
    
    # Scale Regression Features
    reg_features.loc[:, numerical_cols] = scaler.transform(reg_features[numerical_cols])

    # Scale Classification Features
    cls_features.loc[:, numerical_cols] = scaler.transform(cls_features[numerical_cols])

    return reg_features, cls_features


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
    
    # 1. Preprocess the inputs to create both feature vectors
    try:
        reg_features, cls_features = preprocess_input(age, sex, bmi, children, smoker, region, scaler)
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()
        
    
    st.subheader('Prediction Results')
    st.balloons()
    
    
    # --- REGRESSION PREDICTION (8 features) ---
    
    # Make the prediction (log-transformed)
    log_prediction = reg_model.predict(reg_features)
    
    # Inverse transform (exponentiate)
    predicted_charge = np.exp(log_prediction)[0]
    
    
    # --- CLASSIFICATION PREDICTION (6 features) ---
    
    # Make the classification prediction
    predicted_class_encoded = cls_model.predict(cls_features)[0]
    
    # Map the encoded class back to a meaningful category (assuming 0=Low, 1=High)
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
        This value is predicted by the **Stacking Regressor** model (8 features).
        """)
        
    with col_cls:
        st.markdown(f"### Predicted Risk Category (Classification)")
        st.markdown(f"# :{category_color}[{cost_category}]")
        st.markdown(f"""
        This category is predicted by the **Classifier** model (6 features).
        """)
