"""
risk_classifier.py
------------------
Random Forest لتصنيف مناطق الخطر إلى 3 مستويات: low / medium / high
+ SHAP لتفسير قرارات الموديل
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

import sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from config import PROCESSED_DIR, MODELS_DIR, BASE_DIR

log = logging.getLogger(__name__)
REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# Feature Engineering for classification
# ════════════════════════════════════════════════════════════
def build_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # لاغات
    for lag in [1, 2, 3, 6, 12]:
        df[f"twsa_lag{lag}"] = df["twsa_mm"].shift(lag)

    # متوسطات ومعدلات
    for w in [3, 6, 12]:
        df[f"twsa_ma{w}"]  = df["twsa_mm"].rolling(w).mean()
        df[f"twsa_std{w}"] = df["twsa_mm"].rolling(w).std()

    df["twsa_diff1"]    = df["twsa_mm"].diff(1)
    df["twsa_diff3"]    = df["twsa_mm"].diff(3)
    df["twsa_diff12"]   = df["twsa_mm"].diff(12)
    df["trend_slope"]   = df["trend_mm"]

    # مؤشرات هيدرولوجية
    df["water_deficit"] = df["agri_demand_mm"] - df["precipitation_mm"] * 12
    df["precip_anomaly"]= df["precipitation_mm"] - df["precipitation_mm"].rolling(12).mean()

    # دورة موسمية
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["year_norm"] = (df["date"].dt.year - 2002) / 22.0
    df["drought_flag"] = df["drought_flag"].astype(float)

    return df.dropna().reset_index(drop=True)


FEATURE_COLS = [
    "twsa_lag1","twsa_lag2","twsa_lag3","twsa_lag6","twsa_lag12",
    "twsa_ma3","twsa_ma6","twsa_ma12",
    "twsa_std3","twsa_std6",
    "twsa_diff1","twsa_diff3","twsa_diff12",
    "trend_slope","water_deficit","precip_anomaly",
    "precipitation_mm","agri_demand_mm",
    "month_sin","month_cos","year_norm","drought_flag",
]


# ════════════════════════════════════════════════════════════
# تدريب Random Forest
# ════════════════════════════════════════════════════════════
def train_classifier(df_feat: pd.DataFrame):
    avail = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[avail].values.astype(np.float64)
    y = df_feat["risk_level"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)          # high=0, low=1, medium=2

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    # Cross-validation زمنية
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(rf, X, y_enc, cv=cv, scoring="f1_macro")
    log.info(f"  CV F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # تدريب كامل
    rf.fit(X, y_enc)

    # Metrics على كامل البيانات (لعرض الـ confusion matrix)
    y_pred = rf.predict(X)
    y_prob = rf.predict_proba(X)

    report = classification_report(y_enc, y_pred,
                                   target_names=le.classes_, output_dict=True)

    # AUC (one-vs-rest)
    auc = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="macro")
    log.info(f"  Train AUC (OvR): {auc:.3f}")

    # حفظ الموديل
    joblib.dump({"model": rf, "encoder": le, "features": avail},
                MODELS_DIR / "risk_classifier.joblib")

    return rf, le, X, y_enc, y_pred, y_prob, report, auc, avail


# ════════════════════════════════════════════════════════════
# SHAP — تفسير قرارات الموديل
# ════════════════════════════════════════════════════════════
def compute_shap(rf, X, feature_names, n_samples=80):
    log.info(f"  Computing SHAP values on {n_samples} samples...")
    # استخدام TreeExplainer (الأسرع مع Random Forest)
    explainer   = shap.TreeExplainer(rf)
    X_sample    = X[:n_samples]
    sv = explainer.shap_values(X_sample)
    # new shap returns (n, features, classes), old returns list of (n, features)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_values = [sv[:, :, c] for c in range(sv.shape[2])]
    else:
        shap_values = sv
    return shap_values, X_sample


# ════════════════════════════════════════════════════════════
# رسم النتائج
# ════════════════════════════════════════════════════════════
def plot_classifier_results(rf, le, X, y_enc, y_pred, y_prob,
                            shap_values, X_sample, feature_names, auc, cv_scores):
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    RISK_COLORS = {"high": "#E24B4A", "low": "#1D9E75", "medium": "#EF9F27"}
    classes     = le.classes_   # ['high','low','medium']

    # ── 1. Confusion Matrix ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_enc, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                     ha="center", va="center", fontsize=9,
                     color="white" if cm_norm[i,j] > 0.5 else "black")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(classes, fontsize=9)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(classes, fontsize=9)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix\nAUC={auc:.3f}", fontsize=10)
    plt.colorbar(im, ax=ax1, fraction=0.04)

    # ── 2. Feature Importance (RF built-in) ─────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    imp  = rf.feature_importances_
    idx  = np.argsort(imp)[-15:]
    colors_imp = ["#1D6FA5"] * 15
    ax2.barh(range(15), imp[idx], color=colors_imp, alpha=0.8)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax2.set_xlabel("Importance")
    ax2.set_title("Top 15 Feature Importances (Random Forest)", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)

    # ── 3. SHAP Summary (mean |SHAP| per feature) ────────────
    ax3 = fig.add_subplot(gs[1, :2])
    # Mean absolute SHAP across all classes
    mean_shap = np.mean([np.abs(shap_values[c]).mean(axis=0) for c in range(3)], axis=0)
    top_idx   = np.argsort(mean_shap)[-12:]
    ax3.barh(range(12), mean_shap[top_idx], color="#7F77DD", alpha=0.85)
    ax3.set_yticks(range(12))
    ax3.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax3.set_xlabel("Mean |SHAP value|")
    ax3.set_title("SHAP Feature Impact — Top 12 (mean across classes)", fontsize=10)
    ax3.grid(axis="x", alpha=0.25)

    # ── 4. SHAP per class ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    top5 = np.argsort(mean_shap)[-5:]
    x_pos = np.arange(len(classes))
    width = 0.15
    for k, feat_i in enumerate(top5):
        vals = [np.abs(shap_values[c][:, feat_i]).mean() for c in range(3)]
        ax4.bar(x_pos + k*width, vals, width,
                label=feature_names[feat_i][:12], alpha=0.8)
    ax4.set_xticks(x_pos + 2*width)
    ax4.set_xticklabels(classes, fontsize=9)
    ax4.set_title("SHAP per risk class\n(top 5 features)", fontsize=10)
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(axis="y", alpha=0.2)

    # ── 5. Predicted probability over time ───────────────────
    ax5 = fig.add_subplot(gs[2, :])
    colors_cls = [RISK_COLORS[c] for c in classes]
    for c_idx, cls_name in enumerate(classes):
        ax5.plot(y_prob[:, c_idx], lw=1.2,
                 color=RISK_COLORS[cls_name], alpha=0.8, label=f"P({cls_name})")
    ax5.set_xlabel("Month index")
    ax5.set_ylabel("Probability")
    ax5.set_title("Risk Classification Probabilities Over Time", fontsize=10)
    ax5.legend(fontsize=9, loc="upper right")
    ax5.grid(alpha=0.2)
    ax5.set_ylim(0, 1)

    fig.suptitle("Risk Classification — Random Forest + SHAP Explainability",
                 fontsize=13, y=1.01)
    out = REPORTS / "risk_classification.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════
def run_classifier():
    df_raw  = pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])
    df_feat = build_classification_features(df_raw)

    log.info(f"Dataset: {len(df_feat)} samples after feature engineering")
    log.info(f"Risk distribution:\n{df_feat['risk_level'].value_counts().to_string()}")

    rf, le, X, y_enc, y_pred, y_prob, report, auc, features = train_classifier(df_feat)

    shap_vals, X_sample = compute_shap(rf, X, features)

    out = plot_classifier_results(rf, le, X, y_enc, y_pred, y_prob,
                                  shap_vals, X_sample, features, auc,
                                  cv_scores=None)

    return rf, le, shap_vals, features, report, auc, out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    rf, le, shap_vals, features, report, auc, out = run_classifier()
    print(f"\n{'='*52}")
    print(f"  CLASSIFICATION RESULTS")
    print(f"{'='*52}")
    print(f"  AUC (macro OvR): {auc:.4f}")
    print(f"  Classes        : {list(le.classes_)}")
    for cls in le.classes_:
        m = report[cls]
        print(f"  {cls:6s}: precision={m['precision']:.2f}  recall={m['recall']:.2f}  f1={m['f1-score']:.2f}")
    print(f"\n  Chart saved: {out}")
