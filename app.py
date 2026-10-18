import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["ChestPain"] = LabelEncoder().fit(["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    encoders["ExerciseAngina"] = LabelEncoder().fit(["Yes", "No"])
    encoders["Vessels"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Thal"] = LabelEncoder().fit(["Normal", "Fixed Defect", "Reversible Defect"])
    return encoders


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a learning project for the course **Machine Learning Fundamentals (AIC354)**.
    Choose the 4 values below. The trained Support Vector Classifier (SVC)
    will predict if the patient has heart disease.

    *Model accuracy on test data: **85%** (UCI Cleveland dataset, 297 patients).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        index=3,
    )
    exerciseangina = st.selectbox("Chest pain during exercise?", ["No", "Yes"])
with col2:
    vessels = st.selectbox("Number of major vessels (0-3)", ["Zero", "One", "Two", "Three"])
    thal = st.selectbox("Thal (blood test result)", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chestpain],
        "ExerciseAngina": [exerciseangina],
        "Vessels": [vessels],
        "Thal": [thal],
    })

    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "Vessels": encoders["Vessels"].transform(user_input["Vessels"]),
        "Thal": encoders["Thal"].transform(user_input["Thal"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "For learning only. This is NOT a medical tool. Please see a real doctor for any health concern."
)
