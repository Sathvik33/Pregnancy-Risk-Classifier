import streamlit as st
import numpy as np
import joblib
import os

# Load model from 'model' folder
model_path = os.path.join("C:/C_py/Project/Pregnancy risk/Model", "pregnancy_risk_model.pkl")
model = joblib.load(model_path)

st.title("🤰 Pregnancy Risk Classifier")
st.markdown("Enter patient health details below to predict pregnancy risk level.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=60)
temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0)
hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180)
sys_bp = st.number_input("Systolic BP (mm Hg)", min_value=70, max_value=200)
dia_bp = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=130)
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0)
hba1c = st.number_input("Blood Glucose HbA1c (%)", min_value=0, max_value=100)
glucose = st.number_input("Fasting Blood Glucose (mg/dl)", min_value=0.1, max_value=10.0)

# Predict button
if st.button("Predict Risk"):
    input_data = np.array([[age, temp, hr, sys_bp, dia_bp, bmi, hba1c, glucose]])
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("🟢 Low Risk - Pregnancy appears to be progressing well.")
    elif prediction == 1:
        st.warning("🟡 Mid Risk - Please consult a doctor for further assessment.")
    else:
        st.error("🔴 High Risk - Immediate medical attention is recommended.")