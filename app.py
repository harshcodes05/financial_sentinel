import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

st.set_page_config(
    page_title="Financial Sentinel",
    page_icon="💳",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "standard_scaler.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


try:
    rf_model, scaler = load_artifacts()

except Exception as e:
    st.error(f"Failed to load model or scaler.\n\n{e}")
    st.stop()

# Sidebar

st.sidebar.title("💳 Financial Sentinel")

st.sidebar.markdown(
    """
### Credit Card Fraud Detection

This application predicts whether a credit card transaction is **Legitimate** or **Fraudulent** using a trained **Random Forest Classifier**.

The model was trained on the publicly available **Credit Card Fraud Detection** dataset.
"""
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.markdown("""
- **Algorithm:** Random Forest
- **Preprocessing:** StandardScaler
- **Class Imbalance:** SMOTE
- **Features:** 30
""")
st.sidebar.markdown(
    "- **Target:** Binary Classification"
)

st.sidebar.divider()

st.sidebar.info(
    "This application is intended for demonstration and educational purposes."
)

# Main Page

st.title("💳 Financial Sentinel")

st.markdown(
    """
Detect fraudulent credit card transactions using a machine learning model trained on anonymized transaction data.

Enter the transaction details below and click **Predict Transaction**.
"""
)

st.divider()

# User Input

st.header("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Time (seconds since first transaction)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

with col2:
    amount = st.number_input(
        "Transaction Amount (€)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

st.markdown("### PCA Features")

with st.expander("Click to Enter V1 - V28 Features"):

    v_features = []

    cols = st.columns(2)

    for i in range(28):

        with cols[i % 2]:

            value = st.number_input(
                f"V{i+1}",
                value=0.0,
                format="%.6f",
                key=f"V{i+1}"
            )

            v_features.append(value)

# Prediction Pipeline

FEATURE_NAMES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13",
    "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


def predict_transaction(time, amount, v_features):

    # Create feature vector in the same order used during training
    transaction = [time] + v_features + [amount]

    features_df = pd.DataFrame(
        [transaction],
        columns=FEATURE_NAMES
    )

    # Apply the trained StandardScaler
    scaled_features = pd.DataFrame(
        scaler.transform(features_df),
        columns=FEATURE_NAMES
    )

    # Predict class
    prediction = rf_model.predict(scaled_features)[0]

    # Predict probability
    probability = rf_model.predict_proba(scaled_features)[0][1]

    return prediction, probability

# Prediction

st.divider()

predict_button = st.button(
    "🔍 Predict Transaction",
    type="primary",
    use_container_width=True
)

if predict_button:

    prediction, fraud_probability = predict_transaction(
        time,
        amount,
        v_features
    )

    confidence = max(
        fraud_probability,
        1 - fraud_probability
    )

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        if prediction == 1:
            st.error("🚨 Fraudulent")
        else:
            st.success("✅ Legitimate")

    with col2:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    with col3:

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )
    if prediction == 1:

        st.warning(
            """
    This transaction exhibits characteristics similar to
    previous fraudulent transactions.

    Further verification is recommended.
    """
        )

    else:

        st.info(
            """
    This transaction appears to be legitimate based on
    the trained machine learning model.
    """
        )

# Example Transactions

st.divider()

st.header("🧪 Example Transactions")

example_col1, example_col2 = st.columns(2)

with example_col1:

    if st.button("✅ Test Legitimate Transaction"):

        legitimate_time = 0.0

        legitimate_amount = 149.62

        legitimate_v = [
            -1.359807, -0.072781, 2.536347, 1.378155,
            -0.338321, 0.462388, 0.239599, 0.098698,
            0.363787, 0.090794, -0.551600, -0.617801,
            -0.991390, -0.311169, 1.468177, -0.470401,
            0.207971, 0.025791, 0.403993, 0.251412,
            -0.018307, 0.277838, -0.110474, 0.066928,
            0.128539, -0.189115, 0.133558, -0.021053
        ]

        prediction, probability = predict_transaction(
            legitimate_time,
            legitimate_amount,
            legitimate_v
        )
        confidence = max(
            probability,
            1 - probability
        )

        st.success(
            f"Prediction: {'Legitimate' if prediction == 0 else 'Fraudulent'}"
        )

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )
        
with example_col2:

    if st.button("🚨 Test Fraudulent Transaction"):

        fraud_time = 4462.0

        fraud_amount = 239.93

        fraud_v = [
            -2.303350, 1.759247, -0.359745, 2.330243,
            -0.821628, -0.075788, 0.562320, -0.399147,
            -0.238253, -1.525412, 2.032912, -6.560124,
            0.022937, -1.470102, -0.698826, -2.282194,
            -4.781831, -2.615665, -1.334441, -0.430022,
            -0.294166, -0.932391, 0.172726, -0.087330,
            -0.156114, -0.542628, 0.039566, -0.153029
        ]

        prediction, probability = predict_transaction(
            fraud_time,
            fraud_amount,
            fraud_v
        )
        confidence = max(
            probability,
            1 - probability
        )
        st.error(
            f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}"
        )

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )
        
st.divider()

st.caption(
    """
Financial Sentinel

Credit Card Fraud Detection using Machine Learning

Model: Random Forest Classifier

Developed using Python, Scikit-learn and Streamlit.
"""
)