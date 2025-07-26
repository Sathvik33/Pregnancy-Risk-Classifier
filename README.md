# 🤰 Pregnancy Risk Classifier

A machine learning-powered web application that predicts the **risk level** of pregnancy (🟢 Low, 🟡 Mid, 🔴 High) based on critical medical inputs. This project aims to assist in early awareness and care planning for expecting mothers.

---

## 🌟 Inspiration

This project was inspired by an idea I came across while browsing the **Infosys Internship Portal – Projects Section**. The concept of using AI to contribute to **maternal health and safety** deeply resonated with me, motivating the development of a meaningful, life-centric application.

---

## 🚀 Live Demo

👉 [Access the Live App](https://pregnancy-risk-classifier.onrender.com)

> A simple, secure, and responsive web interface that allows users to input medical details and instantly receive pregnancy risk predictions.

---

## 📂 Project Structure


---
Pregnancy-Risk-Classifier/
│
├── Model/
│ └── pregnancy_risk_model.pkl # Trained machine learning model
│
├── p_app.py # Streamlit UI logic
├── README.md # You're here!
└── requirements.txt # Python dependencies
## 🔍 Features

- **Risk Classification**: Categorizes input data into Low, Mid, or High pregnancy risk.
- **Instant Prediction**: Lightweight, responsive Streamlit interface with real-time feedback.
- **User-Friendly UI**: Clean design with intuitive controls.
- **Emoji-Based Labels**: Visual indicators (🟢, 🟡, 🔴) enhance clarity for users.
- **Feedback Styling**: Uses `st.success()` and `st.error()` for visual impact.

---

## 📊 Model Overview

The model is trained on structured medical data and includes features like:
- Age
- Blood Pressure
- Hemoglobin Level
- Diabetes Status
- Heart Rate
- Hypertension
- Previous Pregnancy History
- ... and more (customizable based on dataset)

---

## 🧪 Running Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sathvik33/Pregnancy-Risk-Classifier.git
   cd Pregnancy-Risk-Classifier
