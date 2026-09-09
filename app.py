import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(
    page_title="Gender Classification App",
    page_icon="🧑‍🦱",
    layout="centered"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data['model'], loaded_data['scaler']

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure you have run the Jupyter Notebook to train and save the model first.")
    st.stop()

# Title and description
st.title("🧑‍🦱 Facial Feature Gender Predictor")
st.markdown("Enter the facial features below to predict whether the person is **Male** or **Female**.")
st.divider()

# Input forms
col1, col2 = st.columns(2)

with col1:
    st.subheader("Measurements (cm)")
    forehead_width = st.slider("Forehead Width", min_value=10.0, max_value=16.0, value=13.0, step=0.1, help="Forehead width in cm")
    forehead_height = st.slider("Forehead Height", min_value=4.5, max_value=8.0, value=6.0, step=0.1, help="Forehead height in cm")

with col2:
    st.subheader("Binary Features")
    long_hair = st.radio("Long Hair?", ["Yes", "No"], horizontal=True)
    nose_wide = st.radio("Nose Wide?", ["Yes", "No"], horizontal=True)
    nose_long = st.radio("Nose Long?", ["Yes", "No"], horizontal=True)
    lips_thin = st.radio("Lips Thin?", ["Yes", "No"], horizontal=True)
    distance_nose_to_lip = st.radio("Long Distance from Nose to Lip?", ["Yes", "No"], horizontal=True)

# Map "Yes"/"No" to 1/0
def map_binary(value):
    return 1 if value == "Yes" else 0

# Predict button
if st.button("Predict Gender", type="primary", use_container_width=True):
    # Prepare the input data
    input_data = pd.DataFrame([{
        'long_hair': map_binary(long_hair),
        'forehead_width_cm': forehead_width,
        'forehead_height_cm': forehead_height,
        'nose_wide': map_binary(nose_wide),
        'nose_long': map_binary(nose_long),
        'lips_thin': map_binary(lips_thin),
        'distance_nose_to_lip_long': map_binary(distance_nose_to_lip)
    }])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Extract probabilities mapping to the classes 'Female (0)' and 'Male (1)'
    prob_female = probabilities[0]
    prob_male = probabilities[1]
    
    # Display results
    st.divider()
    
    if prediction == 1:
        st.success(f"### Prediction: Male 👨")
        st.info(f"Confidence: {prob_male * 100:.2f}%")
    else:
        st.success(f"### Prediction: Female 👩")
        st.info(f"Confidence: {prob_female * 100:.2f}%")
