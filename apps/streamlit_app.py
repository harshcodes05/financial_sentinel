import os
import streamlit as st
import requests
import pandas as pd
import time
from pathlib import Path

# Configurable API backend URL with environment variable override
API_URL = os.getenv(
    "FINANCIAL_SENTINEL_API_URL",
    "http://127.0.0.1:8000/api/v1"
)

st.set_page_config(
    page_title="Financial Sentinel | Fraud Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Title Header
st.markdown('<div class="main-header">🛡️ Financial Sentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Credit Card Fraud Detection & Risk Analysis Platform</div>', unsafe_allow_html=True)

# Sidebar System Health & Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/shield-with-authorization-providers.png", width=70)
st.sidebar.title("System Control")

# Check API Health & Latency
api_online = False
latency_ms = 0
try:
    t0 = time.time()
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    latency_ms = (time.time() - t0) * 1000
    if health_resp.status_code == 200:
        api_online = True
        st.sidebar.success(f"🟢 API Backend: Online ({latency_ms:.1f} ms)")
    else:
        st.sidebar.error("🔴 API Backend: Unhealthy")
except Exception:
    st.sidebar.error("🔴 API Backend: Offline (Start `python -m src.api.main`)")

st.sidebar.divider()
st.sidebar.markdown(f"""
**System Specs:**
- **Inference Engine:** FastAPI v2.0
- **Backend Endpoint:** `{API_URL}`
- **Model Stack:** XGBoost v2 + Isolation Forest
- **Optimization:** F2-Tuned Decision Threshold
- **Validation:** Pydantic Strict Schema
""")

# Top Overview KPI Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Backend Status", "ONLINE" if api_online else "OFFLINE", delta="FastAPI REST")
with kpi_col2:
    st.metric("API Latency", f"{latency_ms:.1f} ms" if api_online else "N/A")
with kpi_col3:
    st.metric("Model Architecture", "XGBoost v2", "Dual-Engine")
with kpi_col4:
    st.metric("Security Level", "Strict Validation", "Pydantic v2")

st.divider()

# Navigation Tabs
tab1, tab2 = st.tabs(["⚡ Single Transaction Inspector", "📁 Batch CSV Fraud Inspector"])

# Tab 1: Single Transaction Inspector
with tab1:
    st.subheader("Interactive Transaction Inspector")
    st.caption("Select a preset profile or manually enter feature parameters to analyze fraud risk.")

    # Preset Profile Selector
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    preset_selected = None
    with preset_col1:
        if st.button("🟢 Load Normal Purchase Preset", use_container_width=True):
            preset_selected = "legit"
    with preset_col2:
        if st.button("🟡 Load Medium Risk Preset", use_container_width=True):
            preset_selected = "medium"
    with preset_col3:
        if st.button("🚨 Load High Risk Fraud Preset", use_container_width=True):
            preset_selected = "fraud"

    # Default V Values
    default_legit_v = [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698,
        0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401,
        0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
        0.128539, -0.189115, 0.133558, -0.021053
    ]

    default_fraud_v = [
        -2.303350, 1.759247, -0.359745, 2.330243, -0.821628, -0.075788, 0.562320, -0.399147,
        -0.238253, -1.525412, 2.032912, -6.560124, 0.022937, -1.470102, -0.698826, -2.282194,
        -4.781831, -2.615665, -1.334441, -0.430022, -0.294166, -0.932391, 0.172726, -0.087330,
        -0.156114, -0.542628, 0.039566, -0.153029
    ]

    if preset_selected == "fraud":
        active_time = 4462.0
        active_amount = 239.93
        active_v = default_fraud_v
    elif preset_selected == "medium":
        active_time = 1200.0
        active_amount = 850.00
        active_v = default_legit_v
    else:
        active_time = 0.0
        active_amount = 149.62
        active_v = default_legit_v

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        tx_time = st.number_input("Time Elapsed (Seconds)", min_value=0.0, value=active_time, step=10.0)
    with input_col2:
        tx_amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=active_amount, step=10.0)

    with st.expander("⚙️ Fine-Tune PCA Anonymized Features (V1 to V28)", expanded=False):
        v_features = []
        v_cols = st.columns(4)
        for i in range(28):
            with v_cols[i % 4]:
                v_val = st.number_input(f"V{i+1}", value=active_v[i], format="%.6f", key=f"v_input_{i+1}")
                v_features.append(v_val)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Transaction Risk", type="primary", use_container_width=True)

    if analyze_btn or preset_selected is not None:
        payload = {
            "Time": tx_time,
            "Amount": tx_amount,
            "v_features": v_features
        }

        try:
            start_t = time.time()
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            calc_latency = (time.time() - start_t) * 1000

            if res.status_code == 200:
                data = res.json()
                st.divider()
                st.subheader("📊 Decision & Risk Analysis Report")

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                
                with res_col1:
                    if data["prediction"] == 1:
                        st.error("🚨 FRAUD DETECTED")
                    else:
                        st.success("✅ LEGITIMATE TRANSACTION")

                with res_col2:
                    st.metric("Fraud Probability", f"{data['fraud_probability']:.2%}")
                    st.progress(float(data['fraud_probability']))

                with res_col3:
                    st.metric("Model Confidence", f"{data['confidence']:.2%}")

                with res_col4:
                    risk = data["risk_level"]
                    consensus = data.get("consensus_flag", "N/A")
                    if risk == "HIGH":
                        st.error(f"Risk: {risk} ({consensus})")
                    elif risk == "MEDIUM":
                        st.warning(f"Risk: {risk} ({consensus})")
                    else:
                        st.info(f"Risk: {risk} ({consensus})")

                st.caption(f"⚡ Request completed in {calc_latency:.2f} ms via FastAPI REST API (Dual-Engine XGBoost v2 + Isolation Forest).")
            else:
                st.error(f"API Exception ({res.status_code}): {res.text}")

        except Exception as e:
            st.error(f"Could not connect to FastAPI server: {e}")

# Tab 2: Batch CSV Fraud Inspector (Single High-Throughput REST Batch Request)
with tab2:
    st.subheader("Batch CSV Fraud Detection")
    st.caption("Upload a CSV file containing transaction data to perform bulk inference via the FastAPI `/predict-batch` microservice.")

    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(batch_df):,}** transactions. Preview:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("⚡ Process Batch Predictions", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                total = min(len(batch_df), 100)  # Batch preview window
                status_text.text(f"Assembling batch payload for {total} transactions...")
                progress_bar.progress(0.2)

                # Assemble batch payload array
                batch_payload = []
                for idx in range(total):
                    row = batch_df.iloc[idx]
                    row_time = float(row.get("Time", 0.0))
                    row_amount = float(row.get("Amount", 0.0))
                    row_v = [float(row.get(f"V{i+1}", 0.0)) for i in range(28)]
                    batch_payload.append({
                        "Time": row_time,
                        "Amount": row_amount,
                        "v_features": row_v
                    })

                status_text.text(f"Sending 1 high-performance batch request to {API_URL}/predict-batch...")
                progress_bar.progress(0.6)

                start_batch_t = time.time()
                try:
                    res = requests.post(f"{API_URL}/predict-batch", json=batch_payload, timeout=15)
                    batch_latency = (time.time() - start_batch_t) * 1000

                    if res.status_code == 200:
                        batch_results = res.json()
                        progress_bar.progress(1.0)
                        status_text.success(f"⚡ Batch prediction complete in {batch_latency:.2f} ms via `/predict-batch`!")

                        formatted_results = []
                        for idx, item in enumerate(batch_results):
                            formatted_results.append({
                                "Row": idx + 1,
                                "Amount (€)": batch_payload[idx]["Amount"],
                                "Prediction": item["label"],
                                "Fraud Probability": f"{item['fraud_probability']:.2%}",
                                "Risk Tier": item["risk_level"],
                                "Consensus Flag": item.get("consensus_flag", "CLEAN")
                            })

                        res_df = pd.DataFrame(formatted_results)
                        st.dataframe(res_df, use_container_width=True)

                        fraud_count = len(res_df[res_df["Prediction"] == "Fraudulent"])
                        st.warning(f"Flagged **{fraud_count}** fraudulent transactions out of {total} processed rows.")
                    else:
                        st.error(f"Batch API Error ({res.status_code}): {res.text}")

                except Exception as e:
                    st.error(f"Failed to connect to FastAPI `/predict-batch` endpoint: {e}")

        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
