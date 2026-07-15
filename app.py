import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scalers
rf_model = joblib.load('models/rf_model.pkl')
amount_scaler = joblib.load('models/amount_scaler.pkl')
time_scaler = joblib.load('models/time_scaler.pkl')

FEATURE_NAMES = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18','V19',
                 'V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Amount_scaled','Time_scaled']

st.title('Financial Sentinel')
st.markdown('### Credit Card Fraud Detection')
st.write('Enter transaction details below to check if it is fraudulent.')

st.subheader('Transaction Details')
time = st.number_input('Time (seconds since first transaction)', value=0.0)
amount = st.number_input('Amount (EUR)', value=0.0, min_value=0.0)

st.subheader('PCA Features (V1-V28)')
with st.expander('Click to expand PCA features'):
    v_features = []
    for i in range(1, 29):
        v = st.number_input(f'V{i}', value=0.0)
        v_features.append(v)

def predict(v_vals, time_val, amount_val):
    time_scaled = time_scaler.transform([[time_val]])[0][0]
    amount_scaled = amount_scaler.transform([[amount_val]])[0][0]
    features_df = pd.DataFrame(
        [v_vals + [amount_scaled, time_scaled]],
        columns=FEATURE_NAMES
    )
    return rf_model.predict(features_df)[0]

if st.button('Predict'):
    prediction = predict(v_features, time, amount)
    if prediction == 1:
        st.error('⚠️ FRAUD DETECTED')
    else:
        st.success('✅ Legitimate Transaction')

st.markdown('---')
if st.button('Test with Known Fraud'):
    test_v = [-2.303350, 1.759247, -0.359745, 2.330243, -0.821628,
              -0.075788, 0.562320, -0.399147, -0.238253, -1.525412,
              2.032912, -6.560124, 0.022937, -1.470102, -0.698826,
              -2.282194, -4.781831, -2.615665, -1.334441, -0.430022,
              -0.294166, -0.932391, 0.172726, -0.087330, -0.156114,
              -0.542628, 0.039566, -0.153029]
    prediction = predict(test_v, 4462.0, 239.93)
    if prediction == 1:
        st.error('⚠️ FRAUD DETECTED — Test case confirmed')
    else:
        st.success('✅ Legitimate Transaction')