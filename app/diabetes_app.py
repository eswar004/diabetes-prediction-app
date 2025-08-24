# diabetes_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient details on the left and click **Predict** to estimate diabetes risk using a trained ML model.")

# ---------- Load models ----------
@st.cache_resource
def load_models():
    lr = joblib.load("diabetes_model.pkl")      # Logistic Regression pipeline
    try:
        rf = joblib.load("diabetes_rf.pkl")     # Random Forest 
    except Exception:
        rf = None
    return lr, rf

lr_model, rf_model = load_models()

# ---------- Encoders (must mirror training encodings) ----------
# Explicit categories observed in the dataset
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}

# Update this list to include all categories seen during training
SMOKING_CATS = ["No Info", "current", "ever", "former", "never", "not current"]
SMOKING_CATS = sorted(SMOKING_CATS)  # keep alphabetical for LabelEncoder parity
SMOKING_MAP = {cat: i for i, cat in enumerate(SMOKING_CATS)}

def encode_inputs(gender, smoking_history):
    # fallbacks kept for robustness if a truly unseen category appears
    g = GENDER_MAP.get(gender, GENDER_MAP["Female"])
    if smoking_history not in SMOKING_MAP:
        st.warning(f"Unseen smoking_history '{smoking_history}'—defaulting to 'No Info'. Update SMOKING_CATS if needed.")
    s = SMOKING_MAP.get(smoking_history, SMOKING_MAP["No Info"])
    return g, s

# Feature order must match training exactly
FEATURE_NAMES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
]

# ---------- Sidebar Inputs ----------
st.sidebar.header("Patient Inputs")

gender = st.sidebar.radio("Gender", options=list(GENDER_MAP.keys()), index=0)
age = st.sidebar.slider("Age", min_value=1, max_value=120, value=45, step=1)
hypertension = st.sidebar.selectbox("Hypertension", options=["No", "Yes"], index=0)
heart_disease = st.sidebar.selectbox("Heart Disease", options=["No", "Yes"], index=0)
smoking_history = st.sidebar.selectbox("Smoking History", options=SMOKING_CATS, index=0)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1, format="%.1f")
hba1c = st.sidebar.number_input("HbA1c Level", min_value=3.5, max_value=20.0, value=6.0, step=0.1, format="%.1f")
glucose = st.sidebar.number_input("Blood Glucose (mg/dL)", min_value=40, max_value=500, value=120, step=1)

# Default threshold set via sweep to favor higher recall on LR
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10, max_value=0.90, value=0.45, step=0.01,
    help=(
        "Lower = higher sensitivity (Recall), more positives flagged, more false alarms.\n"
        "Higher = higher specificity/Precision, fewer positives flagged, more missed cases."
    ),
)

model_choice = st.sidebar.selectbox(
    "Model",
    options=["Logistic Regression (balanced)", "Random Forest (balanced)"],
    index=0  # default to LR
)

# ---------- Prepare feature vector ----------
g_code, s_code = encode_inputs(gender, smoking_history)
htn = 1 if hypertension == "Yes" else 0
hd = 1 if heart_disease == "Yes" else 0

x_row = np.array([[g_code, age, htn, hd, s_code, bmi, hba1c, glucose]])
X_df = pd.DataFrame(x_row, columns=FEATURE_NAMES)

# ---------- Prediction ----------
st.subheader("Prediction")

st.markdown(
    """
**Why 0.45 by default?**  
We ran a threshold sweep and picked **0.45** for Logistic Regression to favor **high Recall (~91%)** —  
appropriate for **screening** so we **miss fewer true diabetics**, accepting more false alarms.  
If you want stricter predictions (fewer false positives), raise the threshold or switch to **Random Forest**.
"""
)

if st.button("Predict"):
    model = lr_model if model_choice.startswith("Logistic") else (rf_model or lr_model)
    try:
        proba = float(model.predict_proba(X_df)[:, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred = int(proba >= threshold)
    label = "Likely **Diabetic**" if pred == 1 else "Likely **Not Diabetic**"

    st.markdown(f"### Result: {label}")
    st.metric("Estimated Probability", f"{proba*100:.1f}%")
    st.caption(
        f"Operating point → Model: **{model_choice}**, Threshold: **{threshold:.2f}**  "
        "• Lower threshold → higher Recall • Higher threshold → higher Precision"
    )

    with st.expander("Show input features"):
        st.write(X_df)

st.info(
    "Tip: If you want to **catch more positives**, reduce the threshold (e.g., 0.30–0.40). "
    "If you want **fewer false alarms**, increase it (e.g., 0.60–0.70)."
)

# ---------- Validation Utilities ----------
def load_holdout():
    """Load saved holdout X_test.csv and y_test.csv if present."""
    try:
        Xh = pd.read_csv("X_test.csv")
        yh = pd.read_csv("y_test.csv").squeeze()
        return Xh, yh
    except Exception:
        return None, None

def evaluate_on(dfX, y_true, model, threshold):
    probs = model.predict_proba(dfX)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probs),
        "cm": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4)
    }

# ---------- Validation Expander ----------
with st.expander("🔍 Validation Results (Holdout Set)"):
    st.markdown("Validation metrics on the **saved holdout set** (`X_test.csv` + `y_test.csv`):")

    dfX, y_true = load_holdout()
    if dfX is None:
        st.warning("⚠️ Holdout files not found in working directory. Please place `X_test.csv` and `y_test.csv` alongside the app.")
    else:
        # Ensure correct column order
        missing = set(FEATURE_NAMES) - set(dfX.columns)
        if missing:
            st.error(f"Holdout set is missing required columns: {sorted(missing)}")
        else:
            dfX = dfX[FEATURE_NAMES]

            # Evaluate using the selected model
            model = lr_model if model_choice.startswith("Logistic") else (rf_model or lr_model)
            res = evaluate_on(dfX, y_true, model, threshold)

            st.write({
                "accuracy": round(res["accuracy"], 4),
                "precision": round(res["precision"], 4),
                "recall": round(res["recall"], 4),
                "f1": round(res["f1"], 4),
                "roc_auc": round(res["roc_auc"], 4),
            })

            st.write("Confusion Matrix (rows=true, cols=pred):")
            st.write(pd.DataFrame(res["cm"], index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))

            with st.expander("Classification report"):
                st.text(res["report"])

            # Optional baseline comparison
            try:
                with open("benchmark_metrics.json") as f:
                    base = json.load(f)
                st.caption("Baseline metrics at threshold 0.50 (from training):")
                st.json(base)
            except Exception:
                pass
