import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import torch
import shap
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.pipeline.inference_pipeline import FraudPredictor
from src.models.evaluator import optimize_business_threshold

# Set Matplotlib styles to look beautiful under dark theme
plt.style.use('dark_background')
sns.set_theme(style="dark", palette="muted")

# Unified theme configuration matching the portfolio website
PLOTLY_LAYOUT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#0a0a0a",
    "font": {"color": "white", "family": "DM Sans, sans-serif"}
}

PLOTLY_AXIS_THEME = {
    "gridcolor": "rgba(255, 255, 255, 0.05)",
    "zerolinecolor": "rgba(255, 255, 255, 0.1)",
    "tickfont": {"size": 10},
    "title_font": {"size": 11, "family": "Sora, sans-serif"}
}

# Page configuration for modern premium look
st.set_page_config(
    page_title="Credit Card Fraud Intelligence & Explainability Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark theme and glassmorphism cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Sora:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');
    
    html, body, [class*="css"], .stApp {
        background: radial-gradient(circle at 50% 50%, #110926 0%, #050505 100%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    .main {
        background-color: transparent !important;
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6, [class*="Header"] {
        font-family: 'Sora', 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(8, 8, 8, 0.95) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }
    
    .metric-card {
        background: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 16px;
        padding: 24px 16px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.4s;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 140px;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(168, 85, 247, 0.5) !important;
        box-shadow: 0 12px 30px rgba(168, 85, 247, 0.25);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
        font-family: 'Sora', sans-serif;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Sora', sans-serif;
        line-height: 1.1;
    }
    
    .metric-value.savings {
        color: #00d4ff;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    .metric-value.loss {
        color: #a855f7;
        text-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%) !important;
        color: #050505 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 12px 28px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4) !important;
        color: #050505 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #888888;
        font-size: 1.05rem;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    
    /* Streamlit slider customization */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #00d4ff !important;
        border: 2px solid #a855f7 !important;
        width: 18px !important;
        height: 18px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7) !important;
    }
    
    /* Custom style for numbers inputs, selectors, and dropdowns */
    div[data-baseweb="input"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
    }
    div[data-baseweb="select"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    
    /* Glassmorphic file uploader */
    section[data-testid="stFileUploadDropzone"] {
        background-color: rgba(10, 10, 10, 0.6) !important;
        border: 2px dashed rgba(0, 212, 255, 0.25) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #a855f7 !important;
        background-color: rgba(168, 85, 247, 0.05) !important;
    }
    
    /* Expander visual cleanups */
    div[data-testid="stExpander"] {
        background-color: rgba(10, 10, 10, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Fix side-by-side column header alignment on text wrap */
    .column-header {
        min-height: 56px;
        display: flex;
        align-items: center;
        font-family: 'Sora', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.8rem;
        line-height: 1.35;
    }
</style>
""", unsafe_allow_html=True)

# Title banner
st.markdown('<h1 class="gradient-text" style="font-size: 2.5rem; margin-bottom: 0.2rem;">🛡️ Credit Card Fraud Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.15rem; color: #aaaaaa; font-family: \'Sora\', sans-serif; margin-bottom: 1.5rem;">Production-Grade Stacking Ensemble, Neural Reconstruction Anomaly Engine & Cost-Benefit Optimizer</p>', unsafe_allow_html=True)
st.write("")

# Load model predictor
@st.cache_resource
def get_predictor():
    pipeline_path = "artifacts/full_pipeline.joblib"
    metadata_path = "artifacts/metadata.json"
    if os.path.exists(pipeline_path):
        return FraudPredictor(pipeline_path=pipeline_path, metadata_path=metadata_path)
    return None

predictor = get_predictor()

# Check for test dataset
@st.cache_data
def load_test_data():
    if os.path.exists("data/test.csv"):
        df = pd.read_csv("data/test.csv")
        return df
    return None

test_df = load_test_data()

# Precompute/cache test set predictions for smooth interactive visualization
@st.cache_data
def get_test_predictions():
    if predictor is None or test_df is None:
        return None
    
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].values
    
    # Run prediction
    _, probs = predictor.predict(X_test)
    return {
        "y_true": y_test,
        "y_prob": probs,
        "Amount": X_test["Amount"].values,
        "X_test": X_test
    }

test_preds = get_test_predictions()

# Load metadata
if predictor is not None and os.path.exists("artifacts/metadata.json"):
    with open("artifacts/metadata.json", "r") as f:
        meta = json.load(f)
    optimal_threshold = meta.get("optimal_threshold", 0.1163)
    metrics = meta.get("metrics", {})
    business_metrics = meta.get("business_metrics", {})
else:
    # Demo fallbacks
    optimal_threshold = 0.1163
    metrics = {"auprc": 0.8583, "roc_auc": 0.9765, "precision": 0.6615, "recall": 0.8776, "f1_score": 0.7544}
    business_metrics = {"min_business_cost": 2573.7, "do_nothing_cost": 10644.93, "savings_vs_nothing": 8071.23}

# Sidebar Configurations
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px; padding-top: 10px;">
    <img src="https://img.icons8.com/nolan/256/card-security.png" width="90" style="filter: drop-shadow(0 4px 10px rgba(0, 212, 255, 0.3)); margin-bottom: 10px;">
    <h3 style="font-family: 'Sora', sans-serif; margin-top: 10px; color: #ffffff; font-size: 1.30rem;">Recalibrator Panel</h3>
    <p style="font-size: 0.82rem; color: #888888; font-family: 'DM Sans', sans-serif; line-height: 1.4; padding: 0 10px;">Modify business risk factors dynamically to recalibrate decision thresholds.</p>
</div>
""", unsafe_allow_html=True)
review_cost = st.sidebar.slider("Manual Compliance Review Cost ($)", 0.50, 50.00, 5.00, 0.50)
fraud_factor = st.sidebar.slider("Fraud Loss Factor (Percentage of Amt)", 0.1, 2.0, 1.0, 0.1)

# Helper function to compute costs on the fly
def calculate_interactive_costs(y_true, y_prob, amounts, review_cost, loss_factor):
    thresholds = np.linspace(0.0, 1.0, 100)
    costs = []
    
    pos_mask = (y_true == 1)
    pos_amounts = amounts[pos_mask]
    pos_probs = y_prob[pos_mask]
    
    for t in thresholds:
        flagged = (y_prob >= t).sum()
        fn_cost = pos_amounts[pos_probs < t].sum() * loss_factor
        costs.append(flagged * review_cost + fn_cost)
        
    do_nothing_cost = pos_amounts.sum() * loss_factor
    flag_all_cost = len(y_true) * review_cost
    
    return thresholds, np.array(costs), do_nothing_cost, flag_all_cost

# Live threshold recalculation
if test_preds is not None:
    t_range, cost_vals, do_nothing, flag_all = calculate_interactive_costs(
        test_preds["y_true"],
        test_preds["y_prob"],
        test_preds["Amount"],
        review_cost,
        fraud_factor
    )
    opt_idx = np.argmin(cost_vals)
    tuned_threshold = t_range[opt_idx]
    tuned_cost = cost_vals[opt_idx]
    tuned_savings = do_nothing - tuned_cost
else:
    # Use calibrated standard multiplier for demo values
    tuned_threshold = optimal_threshold
    tuned_cost = business_metrics["min_business_cost"] * (review_cost / 5.0) * fraud_factor
    do_nothing = business_metrics["do_nothing_cost"] * fraud_factor
    tuned_savings = do_nothing - tuned_cost

# Navigation tabs
tab_performance, tab_sandbox, tab_batch = st.tabs([
    "📊 Performance & Live Cost Simulator",
    "🧪 Model Sandbox & Explainability (SHAP)",
    "📂 Batch Prediction Service"
])

# ==================== Tab 1: Performance & Live Cost Simulator ====================
with tab_performance:
    st.markdown("## 📈 Performance Metrics & Financial Calibration")
    st.markdown("Visualize the metrics and adjust compliance values to find the cost-optimal decision threshold.")
    
    # Financial metrics columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Optimal Decision Threshold</div>
            <div class="metric-value">{tuned_threshold:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Do Nothing Cost (Fraud Loss)</div>
            <div class="metric-value">${do_nothing:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Calibrated Cost</div>
            <div class="metric-value">${tuned_cost:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Business Net Savings</div>
            <div class="metric-value savings">${tuned_savings:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    plot_col_left, plot_col_right = st.columns(2)
    
    with plot_col_left:
        st.markdown('<div class="column-header">💼 Business Cost Curve</div>', unsafe_allow_html=True)
        sub_tab_cost = st.tabs(["Cost Calibration Curve"])[0]
        
        if test_preds is not None:
            with sub_tab_cost:
                fig_cost = go.Figure()
                # Cost curve trace - Purple line matching portfolio accent secondary
                fig_cost.add_trace(go.Scatter(
                    x=t_range, y=cost_vals,
                    mode='lines',
                    name='Operational Cost ($)',
                    line=dict(color='#a855f7', width=3)
                ))
                # Optimal marker - Cyan star matching portfolio accent primary
                fig_cost.add_trace(go.Scatter(
                    x=[tuned_threshold], y=[tuned_cost],
                    mode='markers+text',
                    marker=dict(size=14, color='#00d4ff', symbol='star'),
                    text=[f"Optimal: {tuned_threshold:.4f}"],
                    textposition="bottom center",
                    name="Tuned Threshold"
                ))
                # Reference lines
                fig_cost.add_shape(
                    type='line', line=dict(dash='dash', color='#737373', width=1.5),
                    x0=0, x1=1, y0=do_nothing, y1=do_nothing
                )
                fig_cost.add_annotation(
                    x=0.5, y=do_nothing, text="Do Nothing Base Cost", showarrow=False, yshift=10, font=dict(color="#737373")
                )
                
                fig_cost.update_layout(
                    height=450,
                    margin=dict(l=50, r=30, t=30, b=50),
                    xaxis_title="Decision Threshold",
                    yaxis_title="Total Business Cost ($)",
                    **PLOTLY_LAYOUT_THEME
                )
                fig_cost.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_cost.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_cost, use_container_width=True)
        else:
            with sub_tab_cost:
                st.info("Demo Mode: Run training to generate live Business Cost Curve.")
            
    with plot_col_right:
        st.markdown('<div class="column-header">📈 Precision-Recall & ROC Curves</div>', unsafe_allow_html=True)
        if test_preds is not None:
            sub_tab_pr, sub_tab_roc = st.tabs([
                f"Precision-Recall (AUPRC: {metrics['auprc']:.4f})", 
                f"ROC Curve (AUC: {metrics['roc_auc']:.4f})"
            ])
            y_true = test_preds["y_true"]
            y_prob = test_preds["y_prob"]
            
            with sub_tab_pr:
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
                fig_pr = px.line(
                    x=recall, y=precision,
                    labels={"x": "Recall (True Positive Rate)", "y": "Precision (PPV)"}
                )
                fig_pr.update_traces(line=dict(color='#00d4ff', width=3))
                # Highlight tuned threshold - Purple secondary marker
                idx_pr = np.argmin(np.abs(pr_thresholds - tuned_threshold))
                fig_pr.add_trace(go.Scatter(
                    x=[recall[idx_pr]], y=[precision[idx_pr]],
                    mode='markers+text',
                    marker=dict(size=12, color='#a855f7', symbol='circle'),
                    text=[f"Threshold = {tuned_threshold:.4f}"],
                    textposition="top left",
                    name="Tuned Threshold"
                ))
                fig_pr.update_layout(
                    height=450,
                    margin=dict(l=50, r=30, t=30, b=50),
                    **PLOTLY_LAYOUT_THEME
                )
                fig_pr.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_pr.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_pr, use_container_width=True)
                
            with sub_tab_roc:
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                fig_roc = px.line(
                    x=fpr, y=tpr,
                    labels={"x": "False Positive Rate (FPR)", "y": "True Positive Rate (TPR)"}
                )
                fig_roc.update_traces(line=dict(color='#00d4ff', width=3))
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='#737373'), x0=0, x1=1, y0=0, y1=1
                )
                idx_roc = np.argmin(np.abs(roc_thresholds - tuned_threshold))
                fig_roc.add_trace(go.Scatter(
                    x=[fpr[idx_roc]], y=[tpr[idx_roc]],
                    mode='markers+text',
                    marker=dict(size=12, color='#a855f7', symbol='circle'),
                    text=[f"Threshold = {tuned_threshold:.4f}"],
                    textposition="bottom right",
                    name="Tuned Threshold"
                ))
                fig_roc.update_layout(
                    height=450,
                    margin=dict(l=50, r=30, t=30, b=50),
                    **PLOTLY_LAYOUT_THEME
                )
                fig_roc.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_roc.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_roc, use_container_width=True)
        else:
            sub_tab_pr, sub_tab_roc = st.tabs(["Precision-Recall Curve", "ROC Curve"])
            with sub_tab_pr:
                st.info("Demo Mode: Run training to view interactive curves.")

# ==================== Tab 2: Model Sandbox & Explainability with tab_sandbox:
    st.markdown("## 🧪 Live Simulation Sandbox & Model Explanations")
    st.markdown("Score individual transactions, audit their neural reconstruction anomaly, and inspect SHAP attributions.")
    
    # Initialize sandbox inputs state if not present
    if "time_input" not in st.session_state:
        st.session_state.time_input = 3600.0
    if "amount_input" not in st.session_state:
        st.session_state.amount_input = 50.00
    for i in range(1, 29):
        if f"v{i}_input" not in st.session_state:
            st.session_state[f"v{i}_input"] = 0.0

    # Handler for template changes
    def apply_scenario():
        template = st.session_state.scenario_template
        if template == "Legitimate Purchase":
            st.session_state.time_input = 4500.0
            st.session_state.amount_input = 25.50
            for i in range(1, 29):
                st.session_state[f"v{i}_input"] = round(np.random.normal(0.0, 0.15), 4)
        elif template == "High-Value Anomaly (Medium Risk)":
            st.session_state.time_input = 14200.0
            st.session_state.amount_input = 7500.00
            for i in range(1, 29):
                st.session_state[f"v{i}_input"] = round(np.random.normal(0.0, 0.6), 4)
        elif template == "Confirmed Fraud Pattern (High Risk)":
            st.session_state.time_input = 406.0
            st.session_state.amount_input = 239.00
            for i in range(1, 29):
                st.session_state[f"v{i}_input"] = 0.0
            st.session_state.v14_input = -7.3
            st.session_state.v17_input = -6.1
            st.session_state.v12_input = -5.8
            st.session_state.v10_input = -4.5
            st.session_state.v4_input = 4.2
            st.session_state.v11_input = 3.8

    # Scenario selector with template callbacks
    scenario = st.selectbox(
        "Select Scenario Template", 
        [
            "Custom Manual Inputs",
            "Legitimate Purchase",
            "High-Value Anomaly (Medium Risk)",
            "Confirmed Fraud Pattern (High Risk)"
        ],
        key="scenario_template",
        on_change=apply_scenario
    )

    st.markdown("### Transaction Details")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        time_input = st.number_input("Time (Seconds elapsed)", 0.0, 172800.0, key="time_input")
        amount_input = st.number_input("Amount ($)", 0.0, 100000.0, key="amount_input")
    with s_col2:
        st.info("💡 Adjust PCA features V1-V28 below. In production, these represent encrypted transaction embeddings.")
        
    with st.expander("Adjust PCA Transformed Variables (V1 - V28)"):
        v_inputs = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with v_cols[col_idx]:
                v_inputs[f"V{i}"] = st.number_input(f"V{i}", -50.0, 50.0, key=f"v{i}_input")

    # Aggregate inputs
    tx_payload = {"Time": time_input, "Amount": amount_input}
    for i in range(1, 29):
        tx_payload[f"V{i}"] = v_inputs[f"V{i}"]
        
    # Run prediction
    st.write("---")
    st.markdown("### Prediction & Explainability Results")
    
    # Calculate values responsive to inputs
    v14_val = tx_payload["V14"]
    v17_val = tx_payload["V17"]
    v12_val = tx_payload["V12"]
    
    # Sigmoid function based on critical features
    raw_score = -3.5 - 0.7 * v14_val - 0.6 * v17_val - 0.5 * v12_val + 0.00015 * amount_input
    prob = float(np.clip(1.0 / (1.0 + np.exp(-raw_score)), 0.0001, 0.9999))
    recon_err = float(np.clip(0.3 + 0.3 * (v14_val**2 + v17_val**2 + v12_val**2) + 0.00001 * amount_input**2, 0.05, 100.0))
    is_fraud = prob >= tuned_threshold

    if predictor is None:
        st.warning("⚠️ No trained model found. Displaying real-time simulated predictions based on manual inputs.")
        
        res_col_l, res_col_r = st.columns(2)
        with res_col_l:
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Fraud Probability (%)", 'font': {'size': 18, 'family': 'Sora'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#a855f7" if is_fraud else "#00d4ff"},
                    'bgcolor': "#0a0a0a",
                    'borderwidth': 1.5,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, tuned_threshold*100], 'color': "rgba(0, 212, 255, 0.1)"},
                        {'range': [tuned_threshold*100, 100], 'color': "rgba(168, 85, 247, 0.1)"}
                    ]
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white", family="Sora, sans-serif"), height=280)
            st.plotly_chart(fig_g, use_container_width=True)
            
        with res_col_r:
            st.write("### 🚨 System Status & Verdict")
            if is_fraud:
                st.error(f"**DECLINE TRANSACTION**")
                st.write(f"The transaction score is above the cost-optimal decision threshold of **{tuned_threshold:.4f}**.")
            else:
                st.success(f"**APPROVE TRANSACTION**")
                st.write(f"The transaction score is within acceptable bounds below the decision threshold of **{tuned_threshold:.4f}**.")
                
            st.write(f"*   **Simulated Score:** `{prob:.6f}`")
            st.write(f"*   **Simulated Anomaly Score:** `{recon_err:.6f}`")
            
        st.write("---")
        exp_col_l, exp_col_r = st.columns(2)
        
        with exp_col_l:
            st.markdown('<div class="column-header">🔍 Feature Attribution (Simulated SHAP)</div>', unsafe_allow_html=True)
            sub_tab_shap = st.tabs(["SHAP Waterfall Explanation"])[0]
            with sub_tab_shap:
                shap_features = ["Amount", "V14", "V17", "V12", "V10", "V4", "V11", "Time"]
                shap_values = [
                    0.00015 * amount_input,
                    -0.7 * v14_val,
                    -0.6 * v17_val,
                    -0.5 * v12_val,
                    -0.2 * tx_payload.get("V10", 0.0),
                    0.15 * tx_payload.get("V4", 0.0),
                    0.1 * tx_payload.get("V11", 0.0),
                    -0.00001 * time_input
                ]
                df_shap = pd.DataFrame({"Feature": shap_features, "Value": shap_values})
                df_shap = df_shap.sort_values(by="Value", key=abs, ascending=True)
                
                fig_shap = px.bar(
                    df_shap, x="Value", y="Feature",
                    orientation="h",
                    title="Estimated Local Attribution Impact",
                    color="Value",
                    color_continuous_scale=["#a855f7", "#00d4ff"]
                )
                fig_shap.update_layout(
                    height=380,
                    margin=dict(l=60, r=20, t=40, b=40),
                    coloraxis_showscale=False,
                    **PLOTLY_LAYOUT_THEME
                )
                fig_shap.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_shap.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_shap, use_container_width=True)
                
        with exp_col_r:
            st.markdown('<div class="column-header">🧠 Neural Reconstruction Error</div>', unsafe_allow_html=True)
            sub_tab_ae = st.tabs(["Anomaly Driver Decomposition"])[0]
            with sub_tab_ae:
                features = ["V14", "V17", "V12", "V10", "V4", "V11", "V1", "V2", "Amount", "Time"]
                errors = [
                    v14_val**2, v17_val**2, v12_val**2,
                    tx_payload.get("V10", 0.0)**2, tx_payload.get("V4", 0.0)**2,
                    tx_payload.get("V11", 0.0)**2, tx_payload.get("V1", 0.0)**2,
                    tx_payload.get("V2", 0.0)**2, (0.001 * amount_input)**2,
                    (0.0001 * time_input)**2
                ]
                df_ae = pd.DataFrame({"Feature": features, "Squared Error": errors}).sort_values(by="Squared Error", ascending=True)
                
                fig_ae = px.bar(
                    df_ae, x="Squared Error", y="Feature",
                    orientation='h',
                    color="Squared Error",
                    color_continuous_scale=["#00d4ff", "#a855f7"]
                )
                fig_ae.update_layout(
                    height=380,
                    margin=dict(l=60, r=20, t=40, b=40),
                    coloraxis_showscale=False,
                    **PLOTLY_LAYOUT_THEME
                )
                fig_ae.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_ae.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_ae, use_container_width=True)
    else:
        # Step-by-step feature preprocessing to feed SHAP and Autoencoder reconstruction analysis
        cols_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df_tx = pd.DataFrame([tx_payload])[cols_order]
        
        # 1. Transform raw data step-by-step
        X_trans = df_tx.copy()
        tf_steps = ["col_dropper", "time_encoder", "scaler"]
        for name, step in predictor.pipeline.steps:
            if name in tf_steps:
                X_trans = step.transform(X_trans)
                
        # Capture state right before Autoencoder for feature-level reconstruction error
        X_before_ae = X_trans.copy()
        
        # 2. Transform through Autoencoder (appends reconstruction_error)
        autoencoder_transformer = predictor.pipeline.named_steps["autoencoder"]
        X_after_ae = autoencoder_transformer.transform(X_before_ae)
        
        # 3. Score using Ensemble
        stack_clf = predictor.pipeline.named_steps["classifier"]
        prob = stack_clf.predict_proba(X_after_ae)[0, 1]
        is_fraud = prob >= tuned_threshold
        recon_err = X_after_ae["reconstruction_error"].values[0]
        
        res_col_l, res_col_r = st.columns(2)
        
        with res_col_l:
            # Score Gauge
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Fraud Probability (%)", 'font': {'size': 18, 'family': 'Sora'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#a855f7" if is_fraud else "#00d4ff"},
                    'bgcolor': "#0a0a0a",
                    'borderwidth': 1.5,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, tuned_threshold*100], 'color': "rgba(0, 212, 255, 0.1)"},
                        {'range': [tuned_threshold*100, 100], 'color': "rgba(168, 85, 247, 0.1)"}
                    ]
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white", family="Sora, sans-serif"), height=280)
            st.plotly_chart(fig_g, use_container_width=True)
            
        with res_col_r:
            st.write("### 🚨 System Status & Verdict")
            if is_fraud:
                st.error(f"**DECLINE TRANSACTION**")
                st.write(f"The transaction score is above the cost-optimal decision threshold of **{tuned_threshold:.4f}**.")
            else:
                st.success(f"**APPROVE TRANSACTION**")
                st.write(f"The transaction score is within acceptable bounds below the decision threshold of **{tuned_threshold:.4f}**.")
                
            st.write(f"*   **Model Score:** `{prob:.6f}`")
            st.write(f"*   **Neural Reconstruction Error (Anomaly Score):** `{recon_err:.6f}`")
            
        # Explanations layout
        st.write("---")
        exp_col_l, exp_col_r = st.columns(2)
        
        with exp_col_l:
            st.markdown('<div class="column-header">🔍 Feature Attribution (Local SHAP)</div>', unsafe_allow_html=True)
            sub_tab_shap = st.tabs(["SHAP Waterfall Explanation"])[0]
            
            with sub_tab_shap:
                with st.spinner("Computing local SHAP values..."):
                    try:
                        lgb_model = stack_clf.stacking_clf.named_estimators_["lgb"]
                        explainer = shap.TreeExplainer(lgb_model)
                        
                        # Compute SHAP
                        shap_values = explainer(X_after_ae)
                        
                        fig, ax = plt.subplots(figsize=(10, 4.45))
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#050505')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(10)
                            
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not compute SHAP: {e}")
                    
        with exp_col_r:
            st.markdown('<div class="column-header">🧠 Neural Reconstruction Error</div>', unsafe_allow_html=True)
            sub_tab_ae = st.tabs(["Anomaly Driver Decomposition"])[0]
            
            with sub_tab_ae:
                # Compute feature-wise reconstruction error
                ae_net = autoencoder_transformer.model
                ae_net.eval()
                
                columns = list(X_before_ae.columns)
                tensor_x = torch.tensor(X_before_ae.values.astype(np.float32))
                
                with torch.no_grad():
                    reconstructed = ae_net(tensor_x)
                    feature_errors = ((tensor_x - reconstructed) ** 2).numpy()[0]
                    
                df_ae = pd.DataFrame({
                    "Feature": columns,
                    "Squared Error": feature_errors
                }).sort_values(by="Squared Error", ascending=True).tail(10)
                
                fig_ae = px.bar(
                    df_ae, x="Squared Error", y="Feature",
                    orientation='h',
                    color="Squared Error",
                    color_continuous_scale=["#00d4ff", "#a855f7"]
                )
                fig_ae.update_layout(
                    height=400,
                    margin=dict(l=60, r=20, t=10, b=40),
                    coloraxis_showscale=False,
                    **PLOTLY_LAYOUT_THEME
                )
                fig_ae.update_xaxes(**PLOTLY_AXIS_THEME)
                fig_ae.update_yaxes(**PLOTLY_AXIS_THEME)
                st.plotly_chart(fig_ae, use_container_width=True)

# ==================== Tab 3: Batch Prediction Service ====================
with tab_batch:
    st.markdown("## 📂 Batch Scoring Service")
    st.markdown("Upload transactional CSV tables matching the model features schema to compute bulk risk ratings.")
    
    uploaded_file = st.file_uploader("Upload Transaction Table (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"📂 Loaded CSV with {len(batch_df)} rows.")
        st.dataframe(batch_df.head(5))
        
        # Schema verification
        required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        missing = [col for col in required_cols if col not in batch_df.columns]
        
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            if predictor is None:
                st.error("Cannot score batch: Serialized model model files not found.")
            else:
                if st.button("🚀 Process Batch Predictions"):
                    with st.spinner("Scoring batch..."):
                        preds, probs = predictor.predict(batch_df)
                        
                        out_df = batch_df.copy()
                        out_df["Fraud_Probability"] = probs
                        # Mark fraud based on current custom tuned threshold
                        out_df["Is_Fraud_Flag"] = (probs >= tuned_threshold).astype(int)
                        
                        # Statistics
                        total = len(out_df)
                        flagged = out_df["Is_Fraud_Flag"].sum()
                        flag_pct = (flagged / total) * 100
                        
                        st.success("Batch completed successfully!")
                        
                        # Render distribution and overview plots
                        # Summary stats columns (Full-width row)
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Total Records Scored</div>
                                <div class="metric-value">{total}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sc2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Declined Transactions</div>
                                <div class="metric-value loss">{flagged} <span style="font-size: 0.95rem; font-weight: 400; color: #EF4444;">({flag_pct:.2f}%)</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sc3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Approved Transactions</div>
                                <div class="metric-value savings">{total - flagged} <span style="font-size: 0.95rem; font-weight: 400; color: #10B981;">({100 - flag_pct:.2f}%)</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.write("---")
                        
                        # Render distribution and overview plots side-by-side
                        b_col1, b_col2 = st.columns(2)
                        
                        with b_col1:
                            # Interactive Pie Chart
                            fig_pie = px.pie(
                                names=["Approved", "Declined"],
                                values=[total - flagged, flagged],
                                color_discrete_sequence=["#00d4ff", "#a855f7"],
                                title="Transaction Approval Ratio"
                            )
                            fig_pie.update_layout(
                                height=380,
                                margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color="white", family="Sora, sans-serif")
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with b_col2:
                            # Risk score histogram
                            fig_hist = px.histogram(
                                out_df, x="Fraud_Probability",
                                nbins=50,
                                title="Model Fraud Probability Distribution",
                                color_discrete_sequence=["#00d4ff"]
                            )
                            fig_hist.add_vline(x=tuned_threshold, line_dash="dash", line_color="#a855f7", line_width=2)
                            fig_hist.update_layout(
                                height=380,
                                margin=dict(l=20, r=20, t=40, b=20),
                                **PLOTLY_LAYOUT_THEME
                            )
                            fig_hist.update_xaxes(**PLOTLY_AXIS_THEME)
                            fig_hist.update_yaxes(**PLOTLY_AXIS_THEME)
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                        # Download button
                        csv_data = out_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Annotated Predictions CSV",
                            data=csv_data,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
