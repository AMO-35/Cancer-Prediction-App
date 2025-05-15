import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION
st.set_page_config(page_title="Cancer Prediction App", layout="centered")

# HEADER
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🎗️ Cancer Prediction App</h1>
    <p style='text-align: center; color:white;'>Predict whether a person is likely to have cancer based on health and lifestyle factors</p>
    <hr style='border: 1px solid #2E86C1;'>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("User Input Parameters")

Age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50, step=1)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
BMI = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5)
Smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
GeneticRisk = st.sidebar.selectbox("Genetic Risk", ["Low", "Medium", "High"])
PhysicalActivity = st.sidebar.slider("Physical Activity (hours/week)", min_value=0.0, max_value=20.0, value=6.5)
AlcoholIntake = st.sidebar.slider("Alcohol Intake (units/week)", min_value=0.0, max_value=10.0, value=3.2)
CancerHistory = st.sidebar.selectbox("Cancer History in Family", ["No", "Yes"])

# Encode categorical inputs based on dataset
gender_val = 1 if Gender == "Female" else 0
smoking_val = 1 if Smoking == "Yes" else 0
genetic_map = {"Low": 0, "Medium": 1, "High": 2}
genetic_val = genetic_map[GeneticRisk]
cancer_history_val = 1 if CancerHistory == "Yes" else 0

# INPUT DATA
input_data = {
    "Age": Age,
    "Gender": gender_val,
    "BMI": BMI,
    "Smoking": smoking_val,
    "GeneticRisk": genetic_val,
    "PhysicalActivity": PhysicalActivity,
    "AlcoholIntake": AlcoholIntake,
    "CancerHistory": cancer_history_val
}
input_df = pd.DataFrame([input_data])

# LOAD MODEL & PREDICT
model = joblib.load('models/best_model_with_pipeline.pkl')  # Make sure path is correct
result = model.predict(input_df)[0]

# DISPLAY INPUT
st.markdown("""
    <h3 style='text-align: center; color: white;'>User Input Summary</h3>
""", unsafe_allow_html=True)
st.table(input_df)

# RESULT
result_text = "Cancer Detected" if result == 1 else "No Cancer Detected"
result_color = "#E74C3C" if result == 1 else "#27AE60"

st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: {result_color}; color: white; border-radius: 10px;'>
        <h2>{result_text}</h2>
    </div>
""", unsafe_allow_html=True)
