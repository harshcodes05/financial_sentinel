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
- **Inference Engine:** FastAPI
- **Backend Endpoint:** `{API_URL}`
- **Model Stack:** XGBoost + Isolation Forest
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
    st.metric("Model Architecture", "XGBoost", "Dual-Engine")
with kpi_col4:
    st.metric("Security Level", "Strict Validation", "Pydantic v2")

st.divider()

# Navigation Tabs
tab1, tab2 = st.tabs(["⚡ Single Transaction Inspector", "📁 Batch CSV Fraud Inspector"])

# Default V Feature Profiles
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

# Initialize Session State Defaults if Not Present
if "tx_time_val" not in st.session_state:
    st.session_state["tx_time_val"] = 0.0
if "tx_amount_val" not in st.session_state:
    st.session_state["tx_amount_val"] = 149.62
for idx in range(28):
    if f"v_input_{idx+1}" not in st.session_state:
        st.session_state[f"v_input_{idx+1}"] = default_legit_v[idx]

# Tab 1: Single Transaction Inspector
with tab1:
    st.subheader("Interactive Transaction Inspector")
    st.caption("Select a preset profile or manually enter feature parameters to analyze fraud risk.")

    # Preset Profile Selector Buttons
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("🟢 Load Normal Purchase Preset", use_container_width=True):
            st.session_state["tx_time_val"] = 0.0
            st.session_state["tx_amount_val"] = 149.62
            for i in range(28):
                st.session_state[f"v_input_{i+1}"] = default_legit_v[i]
            st.session_state["auto_analyze"] = True
            st.rerun()

    with preset_col2:
        if st.button("🟡 Load Anomaly Alert Preset", use_container_width=True):
            st.session_state["tx_time_val"] = 1200.0
            st.session_state["tx_amount_val"] = 850.00
            for i in range(28):
                st.session_state[f"v_input_{i+1}"] = default_legit_v[i]
            st.session_state["auto_analyze"] = True
            st.rerun()

    with preset_col3:
        if st.button("🚨 Load High Risk Fraud Preset", use_container_width=True):
            st.session_state["tx_time_val"] = 4462.0
            st.session_state["tx_amount_val"] = 239.93
            for i in range(28):
                st.session_state[f"v_input_{i+1}"] = default_fraud_v[i]
            st.session_state["auto_analyze"] = True
            st.rerun()

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        tx_time = st.number_input("Time Elapsed (Seconds)", min_value=0.0, step=10.0, key="tx_time_val")
    with input_col2:
        tx_amount = st.number_input("Transaction Amount (€)", min_value=0.0, step=10.0, key="tx_amount_val")

    with st.expander("⚙️ Fine-Tune PCA Anonymized Features (V1 to V28)", expanded=True):
        v_features = []
        v_cols = st.columns(4)
        for i in range(28):
            with v_cols[i % 4]:
                v_val = st.number_input(f"V{i+1}", format="%.6f", key=f"v_input_{i+1}")
                v_features.append(v_val)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Transaction Risk", type="primary", use_container_width=True)

    should_analyze = analyze_btn or st.session_state.pop("auto_analyze", False)

    if should_analyze:
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

                pred_val = data["prediction"]
                label_str = data["label"]
                prob_val = data["fraud_probability"]
                risk_lvl = data["risk_level"]
                risk_flag = data.get("consensus_flag", "CLEAN")
                is_anom = data.get("is_anomaly", False)

                # Prominent Banner Alert
                if pred_val == 1 and is_anom:
                    st.error(f"### 🚨 CONFIRMED FRAUD DETECTED — Flag: `{risk_flag}` | Risk: `{risk_lvl}`")
                elif pred_val == 1:
                    st.error(f"### 🚨 SUPERVISED FRAUD DETECTED — Flag: `{risk_flag}` | Risk: `{risk_lvl}`")
                elif is_anom:
                    st.warning(f"### ⚠️ UNUSUAL ANOMALY ALERT — Flag: `{risk_flag}` | Risk: `{risk_lvl}`")
                else:
                    st.success(f"### ✅ LEGITIMATE TRANSACTION — Flag: `{risk_flag}` | Risk: `{risk_lvl}`")

                # Prominent Metric Cards
                m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
                
                with m_col1:
                    st.metric("Fraud Probability", f"{prob_val:.2%}")
                    st.progress(float(prob_val))

                with m_col2:
                    st.metric("XGBoost Decision", label_str)

                with m_col3:
                    anom_status = "Anomaly (-1)" if is_anom else "Normal (+1)"
                    st.metric("Isolation Forest", anom_status)

                with m_col4:
                    st.metric("Risk Level", risk_lvl)

                with m_col5:
                    st.metric("Risk Flag", risk_flag)

                with m_col6:
                    st.metric("API Latency", f"{calc_latency:.2f} ms")

                st.caption(f"⚡ Inferred live via FastAPI microservice in {calc_latency:.2f} ms using dual-engine consensus (XGBoost + Isolation Forest).")
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
