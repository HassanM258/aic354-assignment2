"""
Heart Disease Prediction System - Streamlit Web App (AIC354, Assignment 2, Task 2)
-----------------------------------------------------------------------------------
Loads the Support Vector Classifier trained in Heart_Disease_Prediction.ipynb
(svc_trained_model.pkl) and lets a user enter the four input attributes.

It recreates the exact same LabelEncoders used in the notebook (fit on the same
fixed category lists). LabelEncoder assigns codes in sorted order, so the codes
are identical to training - no need to ship the encoders separately.

Adapted from the instructor's Titanic Streamlit demo app.
EDUCATIONAL PROJECT ONLY - not a medical device and not a diagnosis.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Category lists (must match Step 5.1 of the notebook exactly)
# ---------------------------------------------------------------------------
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA = ["Yes", "No"]
MAJOR_VESSELS = ["0 Vessels", "1 Vessel", "2 Vessels", "3 Vessels"]
THALASSEMIA = ["Normal", "Fixed Defect", "Reversible Defect"]

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    return {
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
        "MajorVessels": LabelEncoder().fit(MAJOR_VESSELS),
        "Thalassemia": LabelEncoder().fit(THALASSEMIA),
    }


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Built for **AIC354 – Machine Learning Fundamentals (Assignment 2, Task 2)**.
    Enter a patient's four clinical findings and the trained Support Vector
    Classifier (SVC) predicts whether heart disease is likely.

    *Trained on the UCI Heart Disease (Cleveland) dataset, 297 patients.
    Accuracy on held-out test data: **80%** (5-fold cross-validation: about 77%).*
    """
)
st.warning("Educational project only. This is **not** a medical device and **not** a diagnosis. "
           "See a doctor for any health concern.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
    major_vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", MAJOR_VESSELS)
with col2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA, index=1)
    thalassemia = st.selectbox("Thalassemia (Thallium Stress Test)", THALASSEMIA)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MajorVessels": [major_vessels],
        "Thalassemia": [thalassemia],
    })

    # Same column order as training: ChestPain, ExerciseAngina, MajorVessels, Thalassemia
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ HEART DISEASE NOT LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Predictions reflect patterns in a small, single-hospital dataset (UCI Cleveland). "
    "Only 4 of the 13 clinical attributes are used, as required by the assignment."
)
