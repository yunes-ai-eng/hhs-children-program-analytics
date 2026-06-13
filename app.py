import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HHS AI Analytics Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; }
    .alert-box { padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid; }
    .alert-danger { background: #FEF2F2; border-color: #EF4444; color: #991B1B; }
    .alert-warning { background: #FFFBEB; border-color: #F59E0B; color: #92400E; }
    .alert-success { background: #ECFDF5; border-color: #10B981; color: #065F46; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏛️ HHS Children Program Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Care Transition Efficiency & Placement Outcome Analytics | Unified Mentor</p>', unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("dataset/HHS_Data_Children_Program.csv")
        df.columns = ["date", "apprehended", "in_cbp", "transferred", "in_hhs", "discharged"]
        df["date"] = pd.to_datetime(df["date"])
        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "date": dates,
            "apprehended": np.random.poisson(50, len(dates)),
            "in_cbp": np.random.randint(20, 80, len(dates)),
            "transferred": np.random.poisson(40, len(dates)),
            "in_hhs": np.random.randint(2000, 12000, len(dates)),
            "discharged": np.random.poisson(35, len(dates))
        })
        return df

df = load_data()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df["transfer_efficiency"] = (df["transferred"] / df["in_cbp"].replace(0, np.nan)).clip(0, 1)
df["discharge_effectiveness"] = (
    df["discharged"].rolling(window=30, min_periods=1).sum() /
    df["in_hhs"].replace(0, np.nan)
).clip(0, 1)
df["pipeline_throughput"] = (
    df["discharged"] / (df["apprehended"] + df["transferred"]).replace(0, np.nan)
).clip(0, 1)
df["system_exit_rate"] = (
    (df["transferred"] + df["discharged"]) /
    (df["apprehended"] + df["in_cbp"] + df["in_hhs"]).replace(0, np.nan)
).clip(0, 1)

df["backlog_score"] = df["apprehended"] - df["discharged"]
df["backlog_accumulation_rate"] = df["backlog_score"].diff() / df["apprehended"].replace(0, np.nan)
df["cumulative_backlog"] = df["backlog_score"].cumsum()

df["outcome_stability"] = 1 - df["discharge_effectiveness"].rolling(window=30, min_periods=1).std()
df["outcome_stability"] = df["outcome_stability"].clip(0, 1)

df["inflow"] = df["apprehended"]
df["outflow"] = df["discharged"]
df["net_flow"] = df["inflow"] - df["outflow"]

df["discharge_7d_avg"] = df["discharged"].rolling(window=7, min_periods=1).mean()
df["transfer_7d_avg"] = df["transferred"].rolling(window=7, min_periods=1).mean()

# ============================================================
# SIDEBAR - CONTROL PANEL
# ============================================================
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

# --- Date Filter ---
min_date = df["date"].min().date()
max_date = df["date"].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
filtered_df = df[mask].copy()

st.sidebar.markdown("---")

# --- Forecast Settings ---
st.sidebar.markdown("### 🔮 Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=365, value=90)
forecast_variable = st.sidebar.selectbox(
    "Forecast Variable", ["in_hhs", "in_cbp", "transferred", "discharged", "apprehended"], index=0
)

st.sidebar.markdown("---")

# --- Display Mode ---
st.sidebar.markdown("### 📊 Display Mode")
ratio_mode = st.sidebar.toggle("Show Ratio-Based Metrics", value=False)

# ============================================================
# COMPUTE METRICS (used in Executive Summary + Health Score)
# ============================================================
transfer_score     = filtered_df["transfer_efficiency"].mean() * 100
discharge_score    = filtered_df["discharge_effectiveness"].mean() * 100
throughput_score   = filtered_df["pipeline_throughput"].mean() * 100
stability_score    = filtered_df["outcome_stability"].mean() * 100

health_score = min(round(
    transfer_score * 0.25 +
    discharge_score * 0.25 +
    throughput_score * 0.25 +
    stability_score * 0.25, 2), 100)

latest_data     = filtered_df.iloc[-1] if len(filtered_df) > 0 else df.iloc[-1]
latest_backlog  = filtered_df["backlog_score"].iloc[-1]
latest_efficiency = filtered_df["transfer_efficiency"].iloc[-1]
latest_discharge  = filtered_df["discharge_effectiveness"].iloc[-1]

# Peak HHS load and recent HHS load for summary
hhs_peak  = filtered_df["in_hhs"].max()
hhs_recent = filtered_df["in_hhs"].iloc[-1]
hhs_change_pct = ((hhs_recent - hhs_peak) / hhs_peak * 100) if hhs_peak > 0 else 0

# ============================================================
# EXECUTIVE SUMMARY BUTTON
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Executive Summary")

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if st.sidebar.button(
    "📊 Show Executive Summary",
    key="exec_summary_btn",
    help="Click to view one-page executive summary for stakeholders",
    use_container_width=True,
    type="primary"
):
    st.session_state.show_modal = True

# ============================================================
# EXECUTIVE SUMMARY MODAL (dynamic values)
# ============================================================
if st.session_state.get("show_modal", False):
    health_status = (
        "The system is performing excellently." if health_score >= 80
        else "The system needs attention, primarily in discharge capacity." if health_score >= 60
        else "The system requires immediate intervention."
    )

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem; border-radius: 12px; margin: 1rem 0; color: white;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Executive Summary</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px;
                border: 2px solid #667eea; margin: 1rem 0;">
        <p style="font-size: 1.2rem; color: #374151; text-align: center;">
            <strong>System Health Score: {health_score:.1f}%</strong> — {health_status}
        </p>
        <hr>
        <h3>🎯 Key Findings</h3>
        <ul>
            <li><strong>Transfer Efficiency:</strong> {transfer_score:.1f}% — {'Good CBP→HHS speed' if transfer_score >= 60 else 'Needs improvement'}</li>
            <li><strong>Discharge Effectiveness:</strong> {discharge_score:.1f}% — {'On track' if discharge_score >= 60 else 'Bottleneck identified'}</li>
            <li><strong>Pipeline Throughput:</strong> {throughput_score:.1f}% — {'Overall movement adequate' if throughput_score >= 60 else 'Flow issues detected'}</li>
            <li><strong>Outcome Stability:</strong> {stability_score:.1f}% — {'Highly consistent placements' if stability_score >= 80 else 'Moderate variability'}</li>
            <li><strong>HHS Load Change:</strong> {hhs_change_pct:.1f}% from peak ({hhs_peak:,.0f}) to latest ({hhs_recent:,.0f})</li>
        </ul>
        <hr>
        <h3>💡 Primary Recommendation</h3>
        <p style="background: #ECFDF5; padding: 1rem; border-radius: 8px; border-left: 4px solid #10B981;">
            {'<strong>Maintain current operations</strong> — system is performing well above targets.' if health_score >= 80
             else '<strong>Increase discharge capacity by 40%</strong> to achieve Health Score above 85% and reduce average stay duration.' if health_score >= 60
             else '<strong>Immediate intervention required</strong> — review all pipeline stages and escalate discharge operations.'}
        </p>
        <hr>
        <h3>📈 Selected Period</h3>
        <p>
            Analysis covers <strong>{start_date}</strong> → <strong>{end_date}</strong>
            ({len(filtered_df):,} days of data).
            Current backlog: <strong>{latest_backlog:,.0f}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    close_col1, close_col2, close_col3 = st.columns([1, 2, 1])
    with close_col2:
        if st.button("❌ Close Executive Summary", key="close_modal", use_container_width=True):
            st.session_state.show_modal = False
            st.rerun()

st.sidebar.markdown("---")

# --- Alert Thresholds ---
st.sidebar.markdown("### 🚨 Alert Thresholds")
backlog_threshold = st.sidebar.number_input("Backlog Alert Threshold", value=500, min_value=0)
efficiency_threshold = st.sidebar.slider("Efficiency Alert Threshold (%)", 0, 100, 50) / 100

# ============================================================
# ALERTS
# ============================================================
st.markdown("---")
alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    if latest_backlog > backlog_threshold:
        st.markdown(f'<div class="alert-box alert-danger"><strong>🚨 CRITICAL BACKLOG</strong><br>Backlog: {latest_backlog:,.0f} > {backlog_threshold}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-success"><strong>✅ Backlog Normal</strong><br>Backlog: {latest_backlog:,.0f}</div>', unsafe_allow_html=True)

with alert_col2:
    if latest_efficiency < efficiency_threshold:
        st.markdown(f'<div class="alert-box alert-warning"><strong>⚠️ LOW TRANSFER</strong><br>{latest_efficiency*100:.1f}% < {efficiency_threshold*100:.0f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-success"><strong>✅ Transfer Good</strong><br>{latest_efficiency*100:.1f}%</div>', unsafe_allow_html=True)

with alert_col3:
    if latest_discharge < efficiency_threshold:
        st.markdown(f'<div class="alert-box alert-warning"><strong>⚠️ LOW DISCHARGE</strong><br>{latest_discharge*100:.1f}% < {efficiency_threshold*100:.0f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-success"><strong>✅ Discharge Good</strong><br>{latest_discharge*100:.1f}%</div>', unsafe_allow_html=True)

# ============================================================
# KPIs
# ============================================================
st.markdown("---")
st.subheader("📊 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    if ratio_mode:
        st.metric("Transfer Efficiency", f"{filtered_df['transfer_efficiency'].mean()*100:.1f}%",
                  f"{filtered_df['transfer_efficiency'].iloc[-1]*100:.1f}% today")
    else:
        st.metric("Children in HHS", f"{int(filtered_df['in_hhs'].iloc[-1]):,}",
                  f"{filtered_df['in_hhs'].iloc[-7:].mean():,.0f} 7d avg")

with kpi_col2:
    if ratio_mode:
        st.metric("Discharge Effectiveness", f"{filtered_df['discharge_effectiveness'].mean()*100:.1f}%",
                  f"{filtered_df['discharge_effectiveness'].iloc[-1]*100:.1f}% today")
    else:
        st.metric("Children in CBP", f"{int(filtered_df['in_cbp'].iloc[-1]):,}",
                  f"{filtered_df['in_cbp'].iloc[-7:].mean():,.0f} 7d avg")

with kpi_col3:
    if ratio_mode:
        st.metric("Pipeline Throughput", f"{filtered_df['pipeline_throughput'].mean()*100:.1f}%",
                  f"{filtered_df['pipeline_throughput'].iloc[-1]*100:.1f}% today")
    else:
        st.metric("Daily Transferred", f"{int(filtered_df['transferred'].iloc[-1]):,}",
                  f"{filtered_df['transferred'].iloc[-7:].mean():,.0f} 7d avg")

with kpi_col4:
    if ratio_mode:
        st.metric("Backlog Accum. Rate", f"{filtered_df['backlog_accumulation_rate'].mean():.2f}",
                  f"{filtered_df['backlog_accumulation_rate'].iloc[-1]:.2f} today")
    else:
        st.metric("Daily Discharged", f"{int(filtered_df['discharged'].iloc[-1]):,}",
                  f"{filtered_df['discharged'].iloc[-7:].mean():,.0f} 7d avg")

with kpi_col5:
    if ratio_mode:
        st.metric("Outcome Stability", f"{filtered_df['outcome_stability'].mean()*100:.1f}%",
                  f"{filtered_df['outcome_stability'].iloc[-1]*100:.1f}% today")
    else:
        st.metric("Backlog Score", f"{int(filtered_df['backlog_score'].iloc[-1]):,}",
                  f"{filtered_df['backlog_score'].iloc[-7:].mean():,.0f} 7d avg")

# ============================================================
# SANKEY FLOW
# ============================================================
st.markdown("---")
st.subheader("🔄 Care Pipeline Flow Visualization")

pipeline_col1, pipeline_col2 = st.columns([2, 1])

with pipeline_col1:
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=["Apprehended", "CBP Custody", "HHS Care", "Discharged", "Remaining"],
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            customdata=["New arrivals today", "Current CBP load", "Current HHS load",
                        "Successful placements", "Pending cases"],
            hovertemplate='%{customdata}<br>Count: %{value}<extra></extra>'
        ),
        link=dict(
            source=[0, 1, 2, 2, 1],
            target=[1, 2, 3, 4, 4],
            value=[
                max(int(latest_data["apprehended"]), 1),
                max(int(latest_data["transferred"]), 1),
                max(int(latest_data["discharged"]), 1),
                max(int(latest_data["in_hhs"] - latest_data["discharged"]), 1),
                max(int(latest_data["in_cbp"] - latest_data["transferred"]), 1)
            ],
            customdata=["New intake to CBP", "CBP to HHS transfer", "HHS discharge",
                        "HHS pending cases", "CBP pending cases"],
            hovertemplate='%{customdata}<br>Flow: %{value}<extra></extra>'
        )
    )])
    fig_sankey.update_layout(
        title_text="Daily Pipeline Flow (Hover for details)",
        font_size=12,
        height=400,
        hovermode='closest'
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

with pipeline_col2:
    st.markdown("### 📈 Stage Metrics")
    st.markdown(f"""
    **CBP Stage:**
    - Inflow: {latest_data['apprehended']:,.0f}
    - Current Load: {latest_data['in_cbp']:,.0f}
    - Transferred: {latest_data['transferred']:,.0f}
    - Efficiency: {latest_data['transfer_efficiency']*100:.1f}%

    **HHS Stage:**
    - Current Load: {latest_data['in_hhs']:,.0f}
    - Discharged (30d): {latest_data['discharged']*30:,.0f}
    - Effectiveness: {latest_data['discharge_effectiveness']*100:.1f}%

    **System:**
    - Throughput: {latest_data['pipeline_throughput']*100:.1f}%
    - Net Flow: {latest_data['net_flow']:,.0f}
    """)

# ============================================================
# EFFICIENCY PANELS
# ============================================================
st.markdown("---")
st.subheader("📈 Transfer & Discharge Efficiency Panels")

eff_tab1, eff_tab2, eff_tab3 = st.tabs(["Transfer Efficiency", "Discharge Effectiveness", "Pipeline Throughput"])

with eff_tab1:
    fig_transfer = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Transfer Efficiency (0-100%)", "Daily Volumes"),
                                  vertical_spacing=0.1)
    fig_transfer.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["transfer_efficiency"]*100,
        mode="lines", name="Efficiency %",
        line=dict(color="#667eea", width=2),
        hovertemplate='<b>Transfer Efficiency</b><br>Date: %{x}<br>Efficiency: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)
    fig_transfer.add_hline(y=efficiency_threshold*100, line_dash="dash", line_color="red", row=1, col=1)
    fig_transfer.add_trace(go.Bar(
        x=filtered_df["date"], y=filtered_df["transferred"],
        name="Transferred", marker_color="#4ECDC4",
        hovertemplate='<b>Transferred</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
    ), row=2, col=1)
    fig_transfer.add_trace(go.Bar(
        x=filtered_df["date"], y=filtered_df["in_cbp"],
        name="CBP Load", marker_color="#FF6B6B",
        hovertemplate='<b>CBP Load</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
    ), row=2, col=1)
    fig_transfer.update_layout(height=600, showlegend=True)
    fig_transfer.update_yaxes(range=[0, 100], title_text="Efficiency %", row=1, col=1)
    fig_transfer.update_yaxes(title_text="Count", row=2, col=1)
    st.plotly_chart(fig_transfer, use_container_width=True)

with eff_tab2:
    fig_discharge = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=("Discharge Effectiveness (30-day rolling)", "HHS Load vs Discharged"),
                                   vertical_spacing=0.1)
    fig_discharge.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["discharge_effectiveness"]*100,
        mode="lines", name="Effectiveness %",
        line=dict(color="#10B981", width=2),
        hovertemplate='<b>Discharge Effectiveness</b><br>Date: %{x}<br>Effectiveness: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)
    fig_discharge.add_hline(y=efficiency_threshold*100, line_dash="dash", line_color="red", row=1, col=1)
    fig_discharge.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["in_hhs"],
        mode="lines", name="HHS Load",
        line=dict(color="#F59E0B", width=2),
        hovertemplate='<b>HHS Load</b><br>Date: %{x}<br>Count: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    fig_discharge.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["discharged"].rolling(30).sum(),
        mode="lines", name="Discharged (30d)",
        line=dict(color="#10B981", width=2),
        hovertemplate='<b>30-Day Discharges</b><br>Date: %{x}<br>Count: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    fig_discharge.update_layout(height=600, showlegend=True)
    fig_discharge.update_yaxes(range=[0, 100], title_text="Efficiency %", row=1, col=1)
    st.plotly_chart(fig_discharge, use_container_width=True)

with eff_tab3:
    fig_throughput = go.Figure()
    fig_throughput.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["pipeline_throughput"]*100,
        mode="lines", name="Pipeline Throughput",
        line=dict(color="#8B5CF6", width=2),
        hovertemplate='<b>Pipeline Throughput</b><br>Date: %{x}<br>Throughput: %{y:.1f}%<extra></extra>'
    ))
    fig_throughput.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["system_exit_rate"]*100,
        mode="lines", name="System Exit Rate",
        line=dict(color="#EC4899", width=2, dash="dash"),
        hovertemplate='<b>System Exit Rate</b><br>Date: %{x}<br>Exit Rate: %{y:.1f}%<extra></extra>'
    ))
    fig_throughput.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["outcome_stability"]*100,
        mode="lines", name="Outcome Stability",
        line=dict(color="#10B981", width=2, dash="dot"),
        hovertemplate='<b>Outcome Stability</b><br>Date: %{x}<br>Stability: %{y:.1f}%<extra></extra>'
    ))
    fig_throughput.update_layout(
        title="Pipeline Throughput & System Metrics",
        yaxis_title="Percentage (%)",
        height=500
    )
    fig_throughput.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_throughput, use_container_width=True)

# ============================================================
# BOTTLENECK DETECTION
# ============================================================
st.markdown("---")
st.subheader("🔍 Bottleneck Detection Charts")

bottleneck_col1, bottleneck_col2 = st.columns(2)

with bottleneck_col1:
    fig_backlog = go.Figure()
    fig_backlog.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["cumulative_backlog"],
        mode="lines", name="Cumulative Backlog", fill="tozeroy",
        line=dict(color="#EF4444", width=2),
        hovertemplate='<b>Cumulative Backlog</b><br>Date: %{x}<br>Total: %{y:,.0f}<extra></extra>'
    ))
    fig_backlog.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["backlog_score"],
        mode="lines", name="Daily Backlog",
        line=dict(color="#F59E0B", width=1),
        hovertemplate='<b>Daily Backlog</b><br>Date: %{x}<br>Net: %{y:,.0f}<extra></extra>'
    ))
    fig_backlog.add_hline(y=backlog_threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Alert Threshold ({backlog_threshold})")
    fig_backlog.update_layout(title="Backlog Accumulation Analysis", height=400)
    st.plotly_chart(fig_backlog, use_container_width=True)

with bottleneck_col2:
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=filtered_df["date"], y=filtered_df["inflow"],
        name="Inflow (Apprehended)", marker_color="#3B82F6",
        hovertemplate='<b>Inflow</b><br>Date: %{x}<br>New arrivals: %{y}<extra></extra>'
    ))
    fig_flow.add_trace(go.Bar(
        x=filtered_df["date"], y=filtered_df["outflow"],
        name="Outflow (Discharged)", marker_color="#10B981",
        hovertemplate='<b>Outflow</b><br>Date: %{x}<br>Discharged: %{y}<extra></extra>'
    ))
    fig_flow.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["net_flow"],
        mode="lines", name="Net Flow",
        line=dict(color="#EF4444", width=2),
        hovertemplate='<b>Net Flow</b><br>Date: %{x}<br>Net: %{y:,.0f}<extra></extra>'
    ))
    fig_flow.update_layout(title="Inflow vs Outflow Analysis", barmode="group", height=400)
    st.plotly_chart(fig_flow, use_container_width=True)

st.markdown("### 📊 Stage-wise Bottleneck Analysis")
fig_stages = make_subplots(rows=1, cols=3,
                            subplot_titles=("CBP Load", "HHS Load", "Transfer Queue"),
                            shared_yaxes=True)
fig_stages.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["in_cbp"],
    mode="lines", fill="tozeroy", name="CBP",
    line=dict(color="#4ECDC4"),
    hovertemplate='<b>CBP Load</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
), row=1, col=1)
fig_stages.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["in_hhs"],
    mode="lines", fill="tozeroy", name="HHS",
    line=dict(color="#45B7D1"),
    hovertemplate='<b>HHS Load</b><br>Date: %{x}<br>Count: %{y:,.0f}<extra></extra>'
), row=1, col=2)
fig_stages.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["in_cbp"] - filtered_df["transferred"],
    mode="lines", fill="tozeroy", name="Queue",
    line=dict(color="#FF6B6B"),
    hovertemplate='<b>Transfer Queue</b><br>Date: %{x}<br>Pending: %{y}<extra></extra>'
), row=1, col=3)
fig_stages.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_stages, use_container_width=True)

# ============================================================
# HISTORICAL TRENDS
# ============================================================
st.markdown("---")
st.subheader("📈 Historical Trends")

tabs = st.tabs(["Children in HHS", "Children in CBP", "Apprehended", "Discharged", "All Metrics"])
chart_columns = ["in_hhs", "in_cbp", "apprehended", "discharged"]
chart_labels  = ["Children in HHS", "Children in CBP", "Apprehended", "Discharged"]

for tab, col_name, label in zip(tabs[:4], chart_columns, chart_labels):
    with tab:
        fig = px.line(filtered_df, x="date", y=col_name,
                       title=f"{label} Over Time",
                       labels={col_name: "Count", "date": "Date"})
        fig.add_hline(y=filtered_df[col_name].mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Average: {filtered_df[col_name].mean():,.0f}")
        fig.update_traces(
            hovertemplate=f'<b>{label}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    fig_all = go.Figure()
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
    for col, color, label in zip(chart_columns, colors, chart_labels):
        fig_all.add_trace(go.Scatter(
            x=filtered_df["date"], y=filtered_df[col],
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{label}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
        ))
    fig_all.update_layout(title="All Key Metrics Comparison", height=500, legend=dict(orientation="h"))
    st.plotly_chart(fig_all, use_container_width=True)

# ============================================================
# CORRELATION MATRIX
# ============================================================
st.markdown("---")
st.subheader("🔗 Correlation Matrix")

corr_cols = ["apprehended", "in_cbp", "transferred", "in_hhs", "discharged",
             "transfer_efficiency", "discharge_effectiveness", "pipeline_throughput", "system_exit_rate"]
corr = filtered_df[corr_cols].corr()

fig_corr, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0, fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
plt.title("Correlation Matrix - All Metrics", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
st.pyplot(fig_corr)

st.info("""
💡 **How to read this chart:** Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).
**Red** = strong positive relationship. **Blue** = strong negative relationship.
""")

# ============================================================
# ANOMALY DETECTION  ✅ الإصلاح الرئيسي: list(zip(...))
# ============================================================
st.markdown("---")
st.subheader("🤖 Anomaly Detection")

anomaly_features = ["in_hhs", "in_cbp", "transferred", "discharged"]
anomaly_labels   = ["Children in HHS", "Children in CBP", "Daily Transferred", "Daily Discharged"]

# ✅ الإصلاح: تحويل zip إلى list لتجنب KeyError
selected_anomaly = st.selectbox(
    "Select Variable for Anomaly Detection",
    list(zip(anomaly_features, anomaly_labels)),
    format_func=lambda x: x[1],
    index=0
)
selected_anomaly_col   = selected_anomaly[0]
selected_anomaly_label = selected_anomaly[1]

# ✅ التحقق من وجود العمود قبل الاستخدام
if selected_anomaly_col in filtered_df.columns:
    anomaly_model = IsolationForest(contamination=0.05, random_state=42)
    filtered_df["anomaly_score"] = anomaly_model.fit_predict(filtered_df[[selected_anomaly_col]])
    filtered_df["anomaly"] = filtered_df["anomaly_score"].map({1: "Normal", -1: "Anomaly"})

    fig_anomaly = px.scatter(
        filtered_df, x="date", y=selected_anomaly_col, color="anomaly",
        title=f"Detected Anomalies in {selected_anomaly_label}",
        color_discrete_map={"Normal": "#3B82F6", "Anomaly": "#EF4444"}
    )
    fig_anomaly.update_traces(marker=dict(size=8))
    fig_anomaly.update_layout(height=500, legend_title_text='Status')
    st.plotly_chart(fig_anomaly, use_container_width=True)

    anomaly_count = (filtered_df["anomaly"] == "Anomaly").sum()
    st.info(f"🔍 Detected **{anomaly_count} anomalies** ({anomaly_count/len(filtered_df)*100:.1f}%) in the selected period.")
else:
    st.error(f"Column '{selected_anomaly_col}' not found in data.")

# ============================================================
# AI FORECASTING
# ============================================================
st.markdown("---")
st.subheader("🔮 AI Forecasting Engine")

prophet_df = filtered_df[["date", forecast_variable]].rename(
    columns={"date": "ds", forecast_variable: "y"}
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    growth="flat"
)

if len(prophet_df) > 365:
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

model.fit(prophet_df)

future = model.make_future_dataframe(periods=forecast_days)
future["floor"] = prophet_df["y"].min() * 0.5

forecast = model.predict(future)

min_forecast = max(prophet_df["y"].min() * 0.3, 100)
forecast["yhat"]       = np.maximum(forecast["yhat"], min_forecast)
forecast["yhat_lower"] = np.maximum(forecast["yhat_lower"], min_forecast * 0.5)
forecast["yhat_upper"] = np.maximum(forecast["yhat_upper"], min_forecast)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=prophet_df["ds"], y=prophet_df["y"],
    name="Actual Data", mode="lines",
    line=dict(color="#3B82F6", width=2),
    hovertemplate='<b>Actual Data</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
))
fig_forecast.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    name="Forecast", mode="lines",
    line=dict(color="#10B981", width=2),
    hovertemplate='<b>AI Forecast</b><br>Date: %{x}<br>Predicted: %{y:,.0f}<extra></extra>'
))
fig_forecast.add_trace(go.Scatter(
    x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
    y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"].tolist()[::-1],
    fill="toself", fillcolor="rgba(16, 185, 129, 0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Interval", showlegend=True,
    hovertemplate='<b>Confidence Interval</b><br>Date: %{x}<br>Range: %{y:,.0f}<extra></extra>'
))
fig_forecast.update_layout(
    title=f"AI Forecast for {forecast_variable.replace('_', ' ').title()} ({forecast_days} days)",
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# ============================================================
# AI INSIGHTS
# ============================================================
st.markdown("---")
st.subheader("🧠 AI Insights")

future_value  = forecast["yhat"].iloc[-1]
recent_value  = forecast["yhat"].iloc[-forecast_days] if len(forecast) > forecast_days else forecast["yhat"].iloc[len(forecast)//2]
trend_change  = ((future_value - recent_value) / recent_value * 100) if recent_value > 0 else 0

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    if trend_change > 10:
        st.success(f"📈 Strong Upward Trend\nPredicted: +{trend_change:.1f}%")
    elif trend_change > 0:
        st.info(f"📊 Moderate Increase\nPredicted: +{trend_change:.1f}%")
    elif trend_change > -10:
        st.warning(f"📉 Slight Decline\nPredicted: {trend_change:.1f}%")
    else:
        st.error(f"🚨 Significant Decline\nPredicted: {trend_change:.1f}%")

with insight_col2:
    if "yearly" in forecast.columns and forecast["yearly"].notna().any():
        yearly_peak = forecast.loc[forecast["yearly"].idxmax(), "ds"]
        st.info(f"🌊 Seasonal Pattern\nPeak: {yearly_peak.strftime('%B')}")
    else:
        st.info(f"📅 Forecast Period\n{forecast['ds'].min().strftime('%Y-%m-%d')} → {forecast['ds'].max().strftime('%Y-%m-%d')}")

with insight_col3:
    if forecast["yhat"].iloc[-1] > 0:
        confidence_width = (forecast["yhat_upper"].iloc[-1] - forecast["yhat_lower"].iloc[-1]) / forecast["yhat"].iloc[-1] * 100
        st.success(f"✅ Forecast Confidence\nUncertainty: ±{confidence_width/2:.1f}%")
    else:
        st.warning("⚠️ Low Confidence\nForecast near zero")

# ============================================================
# SYSTEM HEALTH SCORE
# ============================================================
st.markdown("---")
st.subheader("🏥 System Health Score")

health_col1, health_col2 = st.columns([1, 2])

with health_col1:
    st.metric("Overall Health Score", f"{health_score}%")
    st.progress(health_score / 100)
    if health_score >= 80:
        st.success("🟢 System performing excellently")
    elif health_score >= 60:
        st.warning("🟡 System needs attention")
    else:
        st.error("🔴 System requires immediate intervention")

with health_col2:
    fig_health = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Health"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#10B981" if health_score >= 80 else "#F59E0B" if health_score >= 60 else "#EF4444"},
            "steps": [
                {"range": [0, 60],  "color": "#FEF2F2"},
                {"range": [60, 80], "color": "#FFFBEB"},
                {"range": [80, 100],"color": "#ECFDF5"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60
            }
        }
    ))
    fig_health.update_layout(height=300)
    st.plotly_chart(fig_health, use_container_width=True)

st.markdown("### 📊 Component Breakdown")
comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
comp_col1.metric("Transfer Efficiency",     f"{transfer_score:.1f}%")
comp_col2.metric("Discharge Effectiveness", f"{discharge_score:.1f}%")
comp_col3.metric("Pipeline Throughput",     f"{throughput_score:.1f}%")
comp_col4.metric("Outcome Stability",       f"{stability_score:.1f}%")

# ============================================================
# FORECAST TABLE
# ============================================================
st.markdown("---")
st.subheader("📋 Forecast Data")

forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].tail(20).copy()
forecast_display.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound", "Trend"]
forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m-%d")
forecast_display = forecast_display.round(2)
st.dataframe(forecast_display, use_container_width=True)

# ============================================================
# DOWNLOADS
# ============================================================
st.markdown("---")
st.subheader("💾 Export Data")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    csv_buffer = BytesIO()
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_buffer.getvalue(),
        file_name=f"forecast_{forecast_variable}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with download_col2:
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(writer, sheet_name="Forecast", index=False)
        filtered_df.to_excel(writer, sheet_name="Historical Data", index=False)
    st.download_button(
        label="📊 Download Full Excel Report",
        data=excel_buffer.getvalue(),
        file_name=f"hhs_analytics_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with download_col3:
    metrics_buffer = BytesIO()
    metrics_df = pd.DataFrame({
        "Metric": ["Transfer Efficiency", "Discharge Effectiveness", "Pipeline Throughput",
                   "Outcome Stability", "Health Score", "Backlog Score"],
        "Value": [f"{transfer_score:.2f}%", f"{discharge_score:.2f}%", f"{throughput_score:.2f}%",
                  f"{stability_score:.2f}%", f"{health_score:.2f}%", f"{latest_backlog:,.0f}"],
        "Status": [
            "Good" if transfer_score  > 50 else "Warning",
            "Good" if discharge_score > 50 else "Warning",
            "Good" if throughput_score > 50 else "Warning",
            "Good" if stability_score > 50 else "Warning",
            "Good" if health_score    > 60 else "Warning",
            "Good" if latest_backlog  < backlog_threshold else "Critical"
        ]
    })
    metrics_df.to_csv(metrics_buffer, index=False)
    st.download_button(
        label="📈 Download Metrics Summary",
        data=metrics_buffer.getvalue(),
        file_name=f"metrics_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================================
# RAW DATA
# ============================================================
st.markdown("---")
with st.expander("📁 Show Raw Dataset & Engineered Features"):
    display_cols = ["date", "apprehended", "in_cbp", "transferred", "in_hhs", "discharged",
                    "transfer_efficiency", "discharge_effectiveness", "pipeline_throughput",
                    "system_exit_rate", "backlog_score", "backlog_accumulation_rate",
                    "outcome_stability", "anomaly"]
    # anomaly column may not exist if data was too small
    safe_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[safe_cols].round(4), use_container_width=True)
    st.markdown("### 📊 Data Summary Statistics")
    st.dataframe(filtered_df[safe_cols[1:]].describe().round(4), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; padding: 1rem;">
    <p><strong>HHS Children Program Analytics Dashboard</strong> | Unified Mentor Project</p>
    <p>Built with Streamlit, Prophet, Plotly, Scikit-learn</p>
    <p style="font-size: 0.8rem;">Data source: U.S. Department of Health and Human Services</p>
</div>
""", unsafe_allow_html=True)