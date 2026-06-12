"""
eda.py — Exploratory Data Analysis
ينتج 4 مخططات تحليلية ويحفظها كـ PNG + يطبع ملخص إحصائي
"""
import sys
from pathlib import Path
# Ensure project root is on sys.path regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_PYTHON   = _PROJECT_ROOT / "src" / "python"
_SRC_CPP      = _PROJECT_ROOT / "src" / "cpp"
for _p in [str(_SRC_PYTHON), str(_SRC_CPP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PROCESSED_DIR, BASE_DIR

OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        11,
})
COLORS = {"twsa": "#1D6FA5", "precip": "#2D9E75", "demand": "#D85A30", "trend": "#888780"}


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])
    return df


def plot_full_analysis(df):
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── 1. TWSA over time ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(df["date"], df["twsa_mm"], 0,
                     where=(df["twsa_mm"] < 0), alpha=0.25, color=COLORS["twsa"], label="_")
    ax1.fill_between(df["date"], df["twsa_mm"], 0,
                     where=(df["twsa_mm"] >= 0), alpha=0.25, color=COLORS["precip"], label="_")
    ax1.plot(df["date"], df["twsa_mm"], color=COLORS["twsa"], lw=1.3, label="TWSA monthly")
    ax1.plot(df["date"], df["trend_mm"], color="#E24B4A", lw=1.8, ls="--", label="Trend (depletion)")

    droughts = df[df["drought_flag"] == 1]
    ax1.scatter(droughts["date"], droughts["twsa_mm"], color="#D85A30",
                s=22, zorder=5, label="Drought event")

    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.axhline(-50,  color="#EF9F27", lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(-100, color="#E24B4A", lw=0.8, ls="--", alpha=0.7)
    ax1.text(df["date"].iloc[-1], -52,  " Medium risk", color="#EF9F27", fontsize=8, va="bottom")
    ax1.text(df["date"].iloc[-1], -102, " High risk",   color="#E24B4A", fontsize=8, va="bottom")

    ax1.set_title("Terrestrial Water Storage Anomaly — Egypt & North Africa (2002–2023)", fontsize=12)
    ax1.set_ylabel("TWSA (mm)")
    ax1.legend(fontsize=9, loc="upper right")

    # ── 2. هطول الأمطار vs طلب زراعي ─────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(df["date"], df["precipitation_mm"], width=25, color=COLORS["precip"], alpha=0.7, label="Precipitation")
    ax2_r = ax2.twinx()
    ax2_r.plot(df["date"], df["agri_demand_mm"], color=COLORS["demand"], lw=1.2, label="Agri demand")
    ax2_r.spines["top"].set_visible(False)
    ax2.set_title("Precipitation vs Agricultural Demand", fontsize=11)
    ax2.set_ylabel("Precip (mm)", color=COLORS["precip"])
    ax2_r.set_ylabel("Demand (mm)", color=COLORS["demand"])

    # ── 3. توزيع TWSA ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    vals   = df["twsa_mm"].values
    bins   = np.linspace(vals.min() - 5, vals.max() + 5, 35)
    ax3.hist(vals, bins=bins, color=COLORS["twsa"], alpha=0.75, edgecolor="white", lw=0.5)
    ax3.axvline(vals.mean(), color="#E24B4A", lw=1.5, ls="--", label=f"Mean: {vals.mean():.1f} mm")
    ax3.axvline(-50,  color="#EF9F27", lw=1, ls=":", label="Medium threshold")
    ax3.axvline(-100, color="#E24B4A", lw=1, ls=":", label="High threshold")
    ax3.set_title("TWSA Distribution", fontsize=11)
    ax3.set_xlabel("TWSA (mm)")
    ax3.set_ylabel("Frequency (months)")
    ax3.legend(fontsize=8)

    # ── 4. موسمية TWSA ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    df["month"] = df["date"].dt.month
    monthly_mean = df.groupby("month")["twsa_mm"].mean()
    monthly_std  = df.groupby("month")["twsa_mm"].std()
    months_lbl   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax4.bar(range(1,13), monthly_mean, color=COLORS["twsa"], alpha=0.75,
            yerr=monthly_std, capsize=3, ecolor="gray", error_kw={"lw":0.8})
    ax4.set_xticks(range(1,13))
    ax4.set_xticklabels(months_lbl, fontsize=9)
    ax4.set_title("TWSA Seasonal Pattern (mean ± std)", fontsize=11)
    ax4.set_ylabel("TWSA (mm)")

    # ── 5. مستوى الخطر عبر الزمن ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    risk_map  = {"low": 0, "medium": 1, "high": 2}
    risk_colors = {0: COLORS["precip"], 1: "#EF9F27", 2: "#E24B4A"}
    numeric = df["risk_level"].map(risk_map)
    ax5.scatter(df["date"], numeric, c=[risk_colors[v] for v in numeric], s=18, alpha=0.8)
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(["Low", "Medium", "High"])
    ax5.set_title("Risk Level Over Time", fontsize=11)

    fig.suptitle("Water Scarcity Analysis — Egypt & North Africa", fontsize=14, y=1.01)

    out = OUTPUT_DIR / "eda_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def print_summary(df):
    print("\n" + "="*55)
    print("  STATISTICAL SUMMARY")
    print("="*55)
    print(f"  Period      : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total months: {len(df)}")
    print(f"\n  TWSA (mm water equiv.)")
    print(f"    Mean   : {df['twsa_mm'].mean():+.1f}")
    print(f"    Trend  : {df['trend_mm'].iloc[-1]:+.1f} (cumulative depletion)")
    print(f"    Min    : {df['twsa_mm'].min():+.1f}")
    print(f"    Max    : {df['twsa_mm'].max():+.1f}")
    print(f"\n  Risk distribution:")
    for lvl, cnt in df["risk_level"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"    {lvl:6s}: {cnt:3d} months ({pct:4.1f}%) {bar}")
    print(f"\n  Correlation TWSA ↔ Precipitation : {df['twsa_mm'].corr(df['precipitation_mm']):+.3f}")
    print(f"  Correlation TWSA ↔ Agri demand   : {df['twsa_mm'].corr(df['agri_demand_mm']):+.3f}")
    print("="*55)


if __name__ == "__main__":
    df  = load_data()
    out = plot_full_analysis(df)
    print_summary(df)
    print(f"\n  Chart saved: {out}")
