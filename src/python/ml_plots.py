"""
ml_plots.py — رسم نتائج Phase 3 كاملة في لوحة واحدة
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

import sys, logging, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import PROCESSED_DIR, MODELS_DIR, BASE_DIR, RISK_THRESHOLDS

REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def plot_phase3(preds, truth, losses_arr, df_raw):
    train_losses = list(losses_arr[0])
    val_losses   = list(losses_arr[1])

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                          "font.size": 10})

    C = {"pred":"#1D6FA5","truth":"#E24B4A","band":"#BDD6EE",
         "train":"#1D9E75","val":"#EF9F27"}
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── 1. Training curves ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, color=C["train"], lw=1.5, label="Train loss")
    ax1.plot(val_losses,   color=C["val"],   lw=1.5, label="Val loss")
    best_ep = int(np.argmin(val_losses))
    ax1.axvline(best_ep, color="gray", lw=0.8, ls="--")
    ax1.text(best_ep+0.5, max(val_losses)*0.9, f"Best\nep {best_ep}", fontsize=8)
    ax1.set_title("LSTM Learning Curves (Adam)", fontsize=10)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss (scaled)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    # ── 2. Forecast vs truth — all val samples, step +1 ─────
    ax2 = fig.add_subplot(gs[0, 1:])
    n_val = len(preds)
    x_ax  = np.arange(n_val)
    ax2.fill_between(x_ax, truth[:,0], preds[:,0],
                     alpha=0.18, color=C["band"])
    ax2.plot(x_ax, truth[:,0], color=C["truth"], lw=1.4, label="Actual TWSA")
    ax2.plot(x_ax, preds[:,0], color=C["pred"],  lw=1.4, ls="--",
             label="Forecast +1 month")
    ax2.axhline(RISK_THRESHOLDS["low"],    color="#EF9F27", lw=0.8, ls=":", alpha=0.8)
    ax2.axhline(RISK_THRESHOLDS["medium"], color="#E24B4A", lw=0.8, ls=":", alpha=0.8)
    ax2.text(n_val-1, RISK_THRESHOLDS["low"]+1,    " medium risk", fontsize=8, color="#EF9F27")
    ax2.text(n_val-1, RISK_THRESHOLDS["medium"]+1, " high risk",   fontsize=8, color="#E24B4A")
    mae1 = np.abs(preds[:,0]-truth[:,0]).mean()
    ax2.set_title(f"LSTM Forecast vs Actual — Month +1  (MAE={mae1:.1f} mm)", fontsize=10)
    ax2.set_xlabel("Validation sample index"); ax2.set_ylabel("TWSA (mm)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    # ── 3. MAE per horizon step ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mae_steps  = np.abs(preds - truth).mean(axis=0)
    rmse_steps = np.sqrt(((preds-truth)**2).mean(axis=0))
    xh = np.arange(1, 7)
    ax3.bar(xh-0.2, mae_steps,  0.35, color=C["pred"], alpha=0.85, label="MAE")
    ax3.bar(xh+0.2, rmse_steps, 0.35, color=C["truth"],alpha=0.85, label="RMSE")
    ax3.set_xticks(xh)
    ax3.set_xticklabels([f"+{i}mo" for i in range(1,7)])
    ax3.set_title("Forecast Error by Horizon", fontsize=10)
    ax3.set_ylabel("Error (mm)"); ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.2)

    # ── 4. Scatter: predicted vs actual (all horizons) ───────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(truth.ravel(), preds.ravel(), alpha=0.3, s=12, color=C["pred"])
    lo  = min(truth.min(), preds.min()) - 5
    hi  = max(truth.max(), preds.max()) + 5
    ax4.plot([lo,hi],[lo,hi], color="gray", lw=1, ls="--")
    from numpy.polynomial.polynomial import polyfit
    b, a = polyfit(truth.ravel(), preds.ravel(), 1)
    ax4.plot([lo,hi],[b+a*lo,b+a*hi], color=C["truth"], lw=1)
    corr = np.corrcoef(truth.ravel(), preds.ravel())[0,1]
    ax4.set_title(f"Actual vs Predicted  r={corr:.3f}", fontsize=10)
    ax4.set_xlabel("Actual TWSA (mm)"); ax4.set_ylabel("Predicted TWSA (mm)")
    ax4.grid(alpha=0.2)

    # ── 5. 6-month fan forecast from last known point ─────────
    ax5 = fig.add_subplot(gs[1, 2])
    # Use last val sample as example forecast
    last_true = truth[-1]
    last_pred = preds[-1]
    h_ax      = np.arange(1, 7)
    # uncertainty bands (±std of residuals)
    resid_std = np.abs(preds - truth).std(axis=0)
    ax5.fill_between(h_ax, last_pred - resid_std, last_pred + resid_std,
                     alpha=0.2, color=C["pred"], label="±1 std")
    ax5.plot(h_ax, last_true, "o-", color=C["truth"], lw=1.5, ms=6, label="Actual")
    ax5.plot(h_ax, last_pred, "s--",color=C["pred"],  lw=1.5, ms=6, label="Forecast")
    ax5.axhline(RISK_THRESHOLDS["low"],    color="#EF9F27", lw=0.8, ls=":")
    ax5.axhline(RISK_THRESHOLDS["medium"], color="#E24B4A", lw=0.8, ls=":")
    ax5.set_title("6-Month Fan Forecast\n(last val sample)", fontsize=10)
    ax5.set_xlabel("Month ahead"); ax5.set_ylabel("TWSA (mm)")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2)

    # ── 6. Seasonal error pattern ────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    # Align val predictions with dates
    df_dates = df_raw.sort_values("date").reset_index(drop=True)
    # val set starts at 80% of sequences
    seq_len = 12
    offset  = int(len(df_dates) * 0.80) + seq_len
    val_dates = df_dates["date"].iloc[offset:offset+len(preds)].values
    err_step1 = np.abs(preds[:,0] - truth[:,0])
    pd_dates  = pd.DatetimeIndex(val_dates[:len(err_step1)])
    n_common    = min(len(err_step1), len(pd_dates))
    pd_dates_c  = pd_dates[:n_common]
    err_c       = err_step1[:n_common]
    monthly_err = pd.Series(err_c, index=pd_dates_c).groupby(
        pd_dates_c.month).mean()
    mnames = [MONTHS[m-1] for m in monthly_err.index]
    colors_bar = ["#E24B4A" if e > monthly_err.mean() else "#1D6FA5"
                  for e in monthly_err.values]
    ax6.bar(range(len(monthly_err)), monthly_err.values,
            color=colors_bar, alpha=0.85)
    ax6.set_xticks(range(len(monthly_err)))
    ax6.set_xticklabels(mnames, fontsize=9)
    ax6.axhline(monthly_err.mean(), color="gray", lw=1, ls="--",
                label=f"Mean MAE = {monthly_err.mean():.1f} mm")
    ax6.set_title("Forecast Error by Month (seasonal pattern)", fontsize=10)
    ax6.set_ylabel("MAE (mm)"); ax6.legend(fontsize=8); ax6.grid(axis="y",alpha=0.2)

    # ── 7. Model summary box ──────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    overall_mae  = np.abs(preds-truth).mean()
    overall_rmse = np.sqrt(((preds-truth)**2).mean())
    twsa_range   = df_raw["twsa_mm"].max() - df_raw["twsa_mm"].min()
    summary = (
        f"Model: LSTM (NumPy)\n"
        f"Optimizer: Adam\n"
        f"Hidden size: 64\n"
        f"Sequence length: 12 mo\n"
        f"Forecast horizon: 6 mo\n"
        f"Features: 15\n"
        f"Training samples: 188\n"
        f"Val samples: 47\n"
        f"─────────────────────\n"
        f"Overall MAE:  {overall_mae:.2f} mm\n"
        f"Overall RMSE: {overall_rmse:.2f} mm\n"
        f"Normalized MAE: {overall_mae/twsa_range*100:.1f}%\n"
        f"Pearson r: {np.corrcoef(preds.ravel(),truth.ravel())[0,1]:.3f}\n"
        f"─────────────────────\n"
        f"RF Classifier AUC: 0.9998\n"
        f"SHAP: ✓ (TreeExplainer)"
    )
    ax7.text(0.05, 0.97, summary, transform=ax7.transAxes,
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F4F4F5", alpha=0.8))
    ax7.set_title("Phase 3 Summary", fontsize=10)

    fig.suptitle("Phase 3 — ML Models: LSTM Forecasting + Risk Classification",
                 fontsize=13, y=1.01)
    out = REPORTS / "phase3_ml_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    preds  = np.load(MODELS_DIR / "lstm_preds.npy")
    truth  = np.load(MODELS_DIR / "lstm_truth.npy")
    losses = np.load(MODELS_DIR / "lstm_losses.npy", allow_pickle=True)
    df_raw = pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])
    out    = plot_phase3(preds, truth, losses, df_raw)
    print(f"Saved: {out}  ({out.stat().st_size//1024} KB)")
