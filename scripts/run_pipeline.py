"""
run_pipeline.py
---------------
Runs the full analysis pipeline from data generation to ML training.
Execute from the project root:

    python scripts/run_pipeline.py [--steps all|data|simulate|train|dashboard]

Steps:
  data      — Generate synthetic GRACE-style dataset
  simulate  — Run C++ groundwater simulation scenarios
  train     — Train LSTM + Random Forest models
  dashboard — Build the static HTML dashboard
  all       — Run all steps in order (default)
"""
import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "src" / "cpp"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def step_data():
    log.info("═══ STEP 1: Data Generation ═══")
    from data_generator import build_full_dataset, generate_spatial_grid
    df   = build_full_dataset()
    grid = generate_spatial_grid()
    log.info(f"  Dataset : {len(df)} rows × {len(df.columns)} cols")
    log.info(f"  Grid    : {len(grid)} spatial points")


def step_eda():
    log.info("═══ STEP 2: Exploratory Analysis ═══")
    import pandas as pd
    from config import PROCESSED_DIR
    from eda import load_data, plot_full_analysis, print_summary
    df  = load_data()
    out = plot_full_analysis(df)
    print_summary(df)
    log.info(f"  EDA chart saved → {out}")


def step_simulate():
    log.info("═══ STEP 3: C++ Groundwater Simulation ═══")
    try:
        import groundwater_engine  # noqa: F401
    except ImportError:
        log.error("groundwater_engine not compiled. Run: python scripts/build_engine.py")
        sys.exit(1)
    from simulation_runner import build_grid_from_data, run_scenarios, benchmark, plot_simulation_results
    h0 = build_grid_from_data(20, 26)
    results, n_steps = run_scenarios(h0, 20, 26, n_years=20)
    bench = benchmark(h0, 20, 26)
    out   = plot_simulation_results(results, n_steps, bench)
    log.info(f"  Simulation chart → {out}")
    log.info(f"  C++ speedup      : {bench['speedup']:.0f}×")


def step_train():
    log.info("═══ STEP 4: ML Model Training ═══")
    import numpy as np

    log.info("  Training LSTM...")
    from lstm_model import train_lstm
    _, preds, truth, mae_h, rmse_h, info = train_lstm()
    log.info(f"  LSTM MAE  = {info['overall_mae']:.2f} mm")
    log.info(f"  LSTM RMSE = {info['overall_rmse']:.2f} mm")

    log.info("  Training Risk Classifier...")
    from risk_classifier import run_classifier
    _, le, _, _, report, auc, out = run_classifier()
    log.info(f"  RF AUC = {auc:.4f}")
    log.info(f"  Chart  → {out}")

    log.info("  Generating ML plots...")
    from config import MODELS_DIR, PROCESSED_DIR
    import pandas as pd
    from ml_plots import plot_phase3
    pv     = np.load(MODELS_DIR / "lstm_preds.npy")
    tv     = np.load(MODELS_DIR / "lstm_truth.npy")
    losses = np.load(MODELS_DIR / "lstm_losses.npy", allow_pickle=True)
    df_raw = pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])
    out2   = plot_phase3(pv, tv, losses, df_raw)
    log.info(f"  ML chart → {out2}")


def step_dashboard():
    log.info("═══ STEP 5: Dashboard ═══")
    log.info("  To launch the interactive dashboard run:")
    log.info("    streamlit run dashboard/app.py")
    log.info("  Or open dashboard/index.html in any browser (no server needed).")


def main():
    parser = argparse.ArgumentParser(description="Water Scarcity Analysis Pipeline")
    parser.add_argument("--steps", default="all",
                        choices=["all","data","eda","simulate","train","dashboard"])
    args = parser.parse_args()

    log.info("Water Scarcity Analysis Pipeline starting...")
    log.info(f"Project root: {ROOT}")

    step_map = {
        "data":      step_data,
        "eda":       step_eda,
        "simulate":  step_simulate,
        "train":     step_train,
        "dashboard": step_dashboard,
    }

    if args.steps == "all":
        for fn in step_map.values():
            fn()
    else:
        step_map[args.steps]()

    log.info("Pipeline complete ✅")


if __name__ == "__main__":
    main()
