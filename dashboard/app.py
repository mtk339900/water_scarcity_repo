"""
app.py — Water Scarcity Dashboard (Streamlit)
=============================================
شغّل بـ: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_PYTHON   = _PROJECT_ROOT / "src" / "python"
_SRC_CPP      = _PROJECT_ROOT / "src" / "cpp"
for _p in [str(_SRC_PYTHON), str(_SRC_CPP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sys, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import streamlit as st

BASE   = Path(__file__).parent.parent

from config import PROCESSED_DIR, MODELS_DIR, RISK_THRESHOLDS
from lstm_model import WaterLSTM
from risk_classifier import build_classification_features, FEATURE_COLS

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Water Scarcity Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.risk-badge-high   { background:#E24B4A22; color:#E24B4A; border:1px solid #E24B4A; border-radius:6px; padding:2px 10px; font-weight:600; }
.risk-badge-medium { background:#EF9F2722; color:#EF9F27; border:1px solid #EF9F27; border-radius:6px; padding:2px 10px; font-weight:600; }
.risk-badge-low    { background:#1D9E7522; color:#1D9E75; border:1px solid #1D9E75; border-radius:6px; padding:2px 10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"high": "#E24B4A", "medium": "#EF9F27", "low": "#1D9E75"}

# ─── Cached loaders ───────────────────────────────────────────
@st.cache_data
def load_timeseries():
    return pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])

@st.cache_data
def load_grid():
    return pd.read_csv(PROCESSED_DIR / "spatial_grid.csv")

@st.cache_resource
def load_models():
    model = WaterLSTM.load(str(MODELS_DIR / "water_lstm_adam"))
    sc    = pickle.load(open(MODELS_DIR / "lstm_scalers.pkl", "rb"))
    clf   = joblib.load(MODELS_DIR / "risk_classifier.joblib")
    pv    = np.load(MODELS_DIR / "lstm_preds.npy")
    tv    = np.load(MODELS_DIR / "lstm_truth.npy")
    return model, sc, clf, pv, tv

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 Water Scarcity\nMonitor")
    st.markdown("**Egypt & North Africa**")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📈 Time Series",
        "🗺️ Spatial Map",
        "🤖 AI Forecast",
        "⚠️ Risk Alerts",
    ])
    st.markdown("---")
    st.markdown("**Data sources**")
    st.caption("NASA GRACE-FO · ERA5 · FAO")
    st.markdown("---")
    st.caption("Water Scarcity Project · Phase 4")

# ─── Load data ────────────────────────────────────────────────
df   = load_timeseries()
grid = load_grid()
lstm_model, scalers, clf_data, preds_val, truth_val = load_models()
rf, le = clf_data["model"], clf_data["encoder"]

latest      = df.iloc[-1]
latest_twsa = latest["twsa_mm"]
latest_risk = latest["risk_level"]
trend_total = df["trend_mm"].iloc[-1]
n_high      = (df["risk_level"] == "high").sum()
overall_mae = np.abs(preds_val - truth_val).mean()
twsa_range  = df["twsa_mm"].max() - df["twsa_mm"].min()

# ════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💧 Water Scarcity Monitor — Egypt & North Africa")
    st.caption("Powered by NASA GRACE-FO · C++ simulation engine · LSTM + Random Forest")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current TWSA",     f"{latest_twsa:.1f} mm",
              delta=f"{df['twsa_mm'].diff().iloc[-1]:.1f} mm MoM")
    c2.metric("20-yr Depletion",  f"{abs(trend_total):.1f} mm",
              delta="cumulative loss", delta_color="inverse")
    c3.metric("High-Risk Months", f"{n_high} / {len(df)}")
    c4.metric("LSTM MAE",         f"{overall_mae:.1f} mm",
              delta=f"{overall_mae/twsa_range*100:.1f}% of range", delta_color="off")

    st.markdown("---")
    cl, cr = st.columns([2, 1])

    with cl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["twsa_mm"], name="TWSA",
            fill="tozeroy", fillcolor="rgba(29,111,165,0.12)",
            line=dict(color="#1D6FA5", width=1.8)))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["trend_mm"], name="Depletion trend",
            line=dict(color="#E24B4A", width=2, dash="dash")))
        dr = df[df["drought_flag"] == 1]
        fig.add_trace(go.Scatter(
            x=dr["date"], y=dr["twsa_mm"], mode="markers", name="Drought",
            marker=dict(color="#D85A30", size=8, symbol="triangle-down")))
        fig.add_hline(y=RISK_THRESHOLDS["low"],
                      line=dict(color="#EF9F27", dash="dot", width=1))
        fig.add_hline(y=RISK_THRESHOLDS["medium"],
                      line=dict(color="#E24B4A", dash="dot", width=1))
        fig.update_layout(height=320, template="plotly_dark",
                          title="Groundwater Storage Anomaly 2002–2023",
                          yaxis_title="TWSA (mm)",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        rc = df["risk_level"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=rc.index, values=rc.values, hole=0.45,
            marker=dict(colors=[RISK_COLORS[r] for r in rc.index]),
            textinfo="label+percent"))
        fig2.update_layout(height=320, template="plotly_dark",
                           title="Risk Distribution", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏗️ Project Architecture")
    cols = st.columns(4)
    for col, (icon, title, desc, color) in zip(cols, [
        ("📡", "Data Layer",   "NASA GRACE-FO\nERA5 · FAO", "#1D6FA5"),
        ("⚙️", "C++ Engine",  "Darcy FD Sim\n375× faster", "#E24B4A"),
        ("🤖", "ML Models",   "LSTM + RF\nSHAP XAI",      "#7F77DD"),
        ("📊", "Dashboard",   "Streamlit\nPlotly charts", "#1D9E75"),
    ]):
        col.markdown(
            f'<div style="background:{color}1A;border:1px solid {color}66;'
            f'border-radius:10px;padding:14px;text-align:center;">'
            f'<div style="font-size:1.8em">{icon}</div>'
            f'<div style="font-weight:600;margin:6px 0;color:{color}">{title}</div>'
            f'<div style="font-size:0.82em;color:#aaa;white-space:pre-line">{desc}</div>'
            f'</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TIME SERIES
# ════════════════════════════════════════════════════════════
elif page == "📈 Time Series":
    st.title("📈 Groundwater Time Series Analysis")

    cl, cr = st.columns([3, 1])
    with cr:
        show_trend    = st.checkbox("Trend line",      True)
        show_seasonal = st.checkbox("Seasonal cycle",  False)
        show_drought  = st.checkbox("Drought markers", True)
        yr = st.slider("Year range", 2002, 2023, (2002, 2023))

    df_f = df[df["date"].dt.year.between(yr[0], yr[1])]

    with cl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f["date"], y=df_f["twsa_mm"],
            line=dict(color="#1D6FA5", width=1.8), name="TWSA",
            fill="tozeroy", fillcolor="rgba(29,111,165,0.10)"))
        if show_trend:
            fig.add_trace(go.Scatter(x=df_f["date"], y=df_f["trend_mm"],
                line=dict(color="#E24B4A", width=2, dash="dash"), name="Trend"))
        if show_seasonal:
            fig.add_trace(go.Scatter(x=df_f["date"], y=df_f["seasonal_mm"],
                line=dict(color="#1D9E75", width=1.2, dash="dot"), name="Seasonal"))
        if show_drought:
            dr = df_f[df_f["drought_flag"] == 1]
            fig.add_trace(go.Scatter(x=dr["date"], y=dr["twsa_mm"],
                mode="markers", marker=dict(color="#D85A30", size=9, symbol="triangle-down"),
                name="Drought"))
        fig.add_hline(y=RISK_THRESHOLDS["low"],
                      line=dict(color="#EF9F27", dash="dot"),
                      annotation_text="Medium risk threshold")
        fig.add_hline(y=RISK_THRESHOLDS["medium"],
                      line=dict(color="#E24B4A", dash="dot"),
                      annotation_text="High risk threshold")
        fig.update_layout(height=380, template="plotly_dark",
                          yaxis_title="TWSA (mm)", xaxis_title="Date",
                          legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig, use_container_width=True)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df_f["date"], y=df_f["precipitation_mm"],
        name="Precipitation", marker_color="#1D9E75", opacity=0.7), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_f["date"], y=df_f["agri_demand_mm"],
        line=dict(color="#E24B4A", width=1.5), name="Agri demand"), secondary_y=True)
    fig2.update_layout(height=280, template="plotly_dark",
                       title="Precipitation vs Agricultural Demand",
                       legend=dict(orientation="h", y=-0.3))
    fig2.update_yaxes(title_text="Precip (mm)", secondary_y=False)
    fig2.update_yaxes(title_text="Demand (mm)", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Statistics")
    st.dataframe(
        df_f[["twsa_mm","precipitation_mm","agri_demand_mm","water_balance_mm"]]
        .describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════
# SPATIAL MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Spatial Map":
    st.title("🗺️ Spatial Risk Map")

    cl, cr = st.columns([3, 1])
    with cr:
        map_type = st.radio("View", ["TWSA Heatmap", "Risk Categories"])
        st.markdown("---")
        high_pct = (grid["risk"] == "high").mean()   * 100
        med_pct  = (grid["risk"] == "medium").mean() * 100
        low_pct  = 100 - high_pct - med_pct
        st.metric("🔴 High risk",   f"{high_pct:.0f}% of grid")
        st.metric("🟡 Medium risk", f"{med_pct:.0f}% of grid")
        st.metric("🟢 Low risk",    f"{low_pct:.0f}% of grid")

    with cl:
        if map_type == "TWSA Heatmap":
            fig = px.density_heatmap(
                grid, x="longitude", y="latitude", z="twsa_mean",
                nbinsx=13, nbinsy=10,
                color_continuous_scale="RdYlBu",
                labels={"twsa_mean":"TWSA (mm)"},
                title="Mean TWSA — Blue = Surplus · Red = Depletion")
        else:
            fig = px.scatter(
                grid, x="longitude", y="latitude", color="risk",
                color_discrete_map=RISK_COLORS,
                size=[14]*len(grid), size_max=14,
                hover_data={"twsa_mean":":.1f", "risk":True},
                title="Risk Level by Grid Point (10 km resolution)")
        fig.add_shape(type="line", x0=31.5, y0=22, x1=31.5, y1=32,
                      line=dict(color="#4EA8DE", width=2, dash="dot"))
        fig.add_annotation(x=31.7, y=27, text="Nile R.",
                           showarrow=False, font=dict(color="#4EA8DE", size=11))
        fig.update_layout(height=460, template="plotly_dark",
                          xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)")
        st.plotly_chart(fig, use_container_width=True)

    # Year × Month heatmap
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot_table(index="year", columns="month",
                           values="twsa_mm", aggfunc="mean").round(1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig3 = px.imshow(pivot, color_continuous_scale="RdYlBu",
                     labels={"color":"TWSA (mm)","x":"Month","y":"Year"},
                     title="TWSA Calendar Heatmap (year × month)")
    fig3.update_layout(height=380, template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# AI FORECAST
# ════════════════════════════════════════════════════════════
elif page == "🤖 AI Forecast":
    st.title("🤖 LSTM 6-Month Groundwater Forecast")

    cl, cr = st.columns([2, 1])
    with cr:
        st.markdown("### 📐 Model Card")
        st.info(
            f"**Type**: LSTM (pure NumPy)\n\n"
            f"**Optimizer**: Adam (β₁=0.9, β₂=0.999)\n\n"
            f"**Architecture**: 64 → Dense32 → 6\n\n"
            f"**Seq length**: 12 months\n\n"
            f"**Horizon**: 6 months\n\n"
            f"**Features**: 15\n\n"
            f"**Val MAE**: {overall_mae:.2f} mm\n\n"
            f"**Norm. MAE**: {overall_mae/twsa_range*100:.1f}%"
        )
        idx = st.slider("Validation sample", 0, len(preds_val)-1, len(preds_val)-1)

    p         = preds_val[idx]
    t         = truth_val[idx]
    h         = np.arange(1, 7)
    resid_std = np.abs(preds_val - truth_val).std(axis=0)

    with cl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([h, h[::-1]]),
            y=np.concatenate([p+resid_std, (p-resid_std)[::-1]]),
            fill="toself", fillcolor="rgba(29,111,165,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="±1 std"))
        fig.add_trace(go.Scatter(x=h, y=t, mode="lines+markers", name="Actual",
            line=dict(color="#E24B4A", width=2.2),
            marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=h, y=p, mode="lines+markers", name="LSTM Forecast",
            line=dict(color="#1D6FA5", width=2.2, dash="dash"),
            marker=dict(size=9, symbol="square")))
        fig.add_hline(y=RISK_THRESHOLDS["low"],
                      line=dict(color="#EF9F27", dash="dot"),
                      annotation_text="Medium risk")
        fig.add_hline(y=RISK_THRESHOLDS["medium"],
                      line=dict(color="#E24B4A", dash="dot"),
                      annotation_text="High risk")
        fig.update_layout(
            title=f"6-Month Forecast vs Actual — Sample #{idx}  "
                  f"(MAE = {np.abs(p-t).mean():.1f} mm)",
            xaxis_title="Month ahead", yaxis_title="TWSA (mm)",
            height=370, template="plotly_dark",
            legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        mae_h  = np.abs(preds_val - truth_val).mean(axis=0)
        rmse_h = np.sqrt(((preds_val - truth_val)**2).mean(axis=0))
        fig2   = go.Figure()
        fig2.add_trace(go.Bar(x=[f"+{i}mo" for i in range(1,7)], y=mae_h,
            name="MAE", marker_color="#1D6FA5"))
        fig2.add_trace(go.Bar(x=[f"+{i}mo" for i in range(1,7)], y=rmse_h,
            name="RMSE", marker_color="#E24B4A", opacity=0.8))
        fig2.update_layout(barmode="group", height=280, template="plotly_dark",
                           title="Error by Horizon Step",
                           yaxis_title="mm", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        corr = np.corrcoef(preds_val.ravel(), truth_val.ravel())[0,1]
        fig3 = go.Figure()
        lo = min(truth_val.min(), preds_val.min()) - 5
        hi = max(truth_val.max(), preds_val.max()) + 5
        fig3.add_trace(go.Scatter(
            x=truth_val.ravel(), y=preds_val.ravel(), mode="markers",
            marker=dict(color="#7F77DD", size=5, opacity=0.55), name="Samples"))
        fig3.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi],
            line=dict(color="gray", dash="dash"), name="Perfect"))
        fig3.update_layout(height=280, template="plotly_dark",
                           title=f"Actual vs Predicted  r={corr:.3f}",
                           xaxis_title="Actual (mm)", yaxis_title="Predicted (mm)")
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# RISK ALERTS
# ════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Alerts":
    st.title("⚠️ Risk Alert System")

    if latest_risk == "high":
        st.error(f"🚨 CRITICAL — TWSA = {latest_twsa:.1f} mm | HIGH RISK")
    elif latest_risk == "medium":
        st.warning(f"⚠️ WARNING — TWSA = {latest_twsa:.1f} mm | MEDIUM RISK")
    else:
        st.success(f"✅ STABLE — TWSA = {latest_twsa:.1f} mm | LOW RISK")

    st.markdown("---")

    # 6-month forecast risk table
    st.markdown("### 🔮 6-Month Outlook (LSTM)")
    last_pred = preds_val[-1]
    resid_std = np.abs(preds_val - truth_val).std(axis=0)
    rows = []
    for i, (pred, std) in enumerate(zip(last_pred, resid_std)):
        risk = ("high"   if pred < RISK_THRESHOLDS["medium"] else
                "medium" if pred < RISK_THRESHOLDS["low"]    else "low")
        rows.append({
            "Horizon":    f"Month +{i+1}",
            "TWSA (mm)":  round(float(pred), 1),
            "Uncert. ±":  round(float(std), 1),
            "Risk":       risk.upper(),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Horizon"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📅 High-Risk Months (2002–2023)")
        hm = df[df["risk_level"]=="high"][["date","twsa_mm","drought_flag"]].copy()
        hm["drought_flag"] = hm["drought_flag"].map({1:"🔴 Yes", 0:"No"})
        hm.columns = ["Date","TWSA (mm)","Drought"]
        st.dataframe(hm.set_index("Date"), use_container_width=True)

    with c2:
        st.markdown("### 📉 Annual Depletion Rate")
        dep_rate = abs(trend_total) / 22
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dep_rate,
            delta={"reference": 4.0, "increasing": {"color":"#E24B4A"}},
            title={"text": "mm / year"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar":  {"color": "#1D6FA5"},
                "steps": [
                    {"range": [0, 3], "color": "#1D9E75"},
                    {"range": [3, 6], "color": "#EF9F27"},
                    {"range": [6,10], "color": "#E24B4A"},
                ],
                "threshold": {"line":{"color":"white","width":3},
                              "thickness":0.75,"value":dep_rate},
            }
        ))
        fig.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Risk transitions sunburst
    st.markdown("### 🔄 Risk Transitions")
    df["risk_next"] = df["risk_level"].shift(-1)
    tr = df.groupby(["risk_level","risk_next"]).size().reset_index(name="n").dropna()
    fig2 = px.sunburst(tr, path=["risk_level","risk_next"], values="n",
                       color="risk_level", color_discrete_map=RISK_COLORS,
                       title="Month-to-month risk level transitions")
    fig2.update_layout(height=340, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)
