"""
data_generator.py — ينتج بيانات واقعية محاكاة لمنطقة مصر وشمال أفريقيا
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

import logging
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, REGION, TIME_SERIES, RISK_THRESHOLDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def generate_grace_timeseries(seed=42):
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(f"{TIME_SERIES['start_year']}-01-01",
                          f"{TIME_SERIES['end_year']}-12-01", freq="MS")
    n = len(dates)

    trend    = np.linspace(0, -5.0 * (TIME_SERIES["end_year"] - TIME_SERIES["start_year"]), n)
    t        = np.linspace(0, 2 * np.pi * n / 12, n)
    seasonal = 8.0 * np.sin(t) + 3.5 * np.sin(2 * t + 0.5)
    noise    = rng.normal(0, 6.0, n)

    years  = dates.year.to_numpy()
    months = dates.month.to_numpy()
    drought_mask = (
        ((years == 2011) & np.isin(months, [6,7,8,9])) |
        ((years == 2012) & np.isin(months, [1,2,3]))   |
        ((years == 2017) & np.isin(months, [3,4,5,6])) |
        ((years == 2021) & np.isin(months, [5,6,7,8]))
    )
    drought_impact = np.where(drought_mask, rng.uniform(-25, -15, n), 0)
    twsa = trend + seasonal + noise + drought_impact

    log.info(f"TWSA generated: {n} months | range {twsa.min():.1f} to {twsa.max():.1f} mm")
    return pd.DataFrame({
        "date":         dates,
        "twsa_mm":      np.round(twsa, 2),
        "trend_mm":     np.round(trend, 2),
        "seasonal_mm":  np.round(seasonal, 2),
        "drought_flag": drought_mask.astype(int),
    })


def generate_precipitation(dates_index, seed=99):
    rng    = np.random.default_rng(seed)
    months = pd.DatetimeIndex(dates_index).month.to_numpy()
    n      = len(months)

    base   = np.where(np.isin(months, [10,11,12,1,2]), 18.0, 2.0)
    noise  = rng.exponential(3.0, n)
    trend  = np.linspace(0, -3.0, n)
    precip = np.maximum(base + noise + trend, 0.0)

    log.info(f"Precipitation generated: mean={precip.mean():.1f} mm/month")
    return np.round(precip, 2)


def generate_agricultural_demand(dates_index, seed=77):
    rng    = np.random.default_rng(seed)
    months = pd.DatetimeIndex(dates_index).month.to_numpy()
    n      = len(months)

    base   = np.where(np.isin(months, [3,4,5,6,7,8]), 180.0, 90.0)
    trend  = np.linspace(0, 25.0, n)
    noise  = rng.normal(0, 8.0, n)
    demand = base + trend + noise

    log.info(f"Agricultural demand generated: mean={demand.mean():.1f} mm/month")
    return np.round(demand, 2)


def build_full_dataset():
    log.info("Building full dataset...")
    df = generate_grace_timeseries()

    df["precipitation_mm"] = generate_precipitation(df["date"])
    df["agri_demand_mm"]   = generate_agricultural_demand(df["date"])
    df["water_balance_mm"] = (df["precipitation_mm"] - df["agri_demand_mm"] / 12).round(2)
    df["cumulative_deficit"] = df["twsa_mm"].cumsum().round(2)

    def classify_risk(v):
        if v > RISK_THRESHOLDS["low"]:    return "low"
        if v > RISK_THRESHOLDS["medium"]: return "medium"
        return "high"

    df["risk_level"] = df["twsa_mm"].apply(classify_risk)

    out = PROCESSED_DIR / "water_data_full.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved: {out} | {len(df)} rows x {len(df.columns)} cols")
    return df


def generate_spatial_grid(n_lat=10, n_lon=13, seed=55):
    rng  = np.random.default_rng(seed)
    lats = np.linspace(REGION["lat_min"], REGION["lat_max"], n_lat)
    lons = np.linspace(REGION["lon_min"], REGION["lon_max"], n_lon)

    rows = []
    for lat in lats:
        for lon in lons:
            nile_proximity = 1 - abs(lon - 31.5) / 15
            nubian_effect  = -30 if (lat < 26 and lon < 32) else 0
            twsa_mean = -80 + nile_proximity * 20 + nubian_effect + rng.normal(0, 12)
            rows.append({
                "latitude":  round(lat, 2),
                "longitude": round(lon, 2),
                "twsa_mean": round(twsa_mean, 1),
                "risk": "high" if twsa_mean < -100 else "medium" if twsa_mean < -50 else "low",
            })

    grid = pd.DataFrame(rows)
    out  = PROCESSED_DIR / "spatial_grid.csv"
    grid.to_csv(out, index=False)
    log.info(f"Spatial grid saved: {out} | {len(grid)} points")
    return grid


if __name__ == "__main__":
    df   = build_full_dataset()
    grid = generate_spatial_grid()

    print("\n--- عينة من البيانات ---")
    print(df[["date","twsa_mm","precipitation_mm","agri_demand_mm","risk_level"]].head(10).to_string(index=False))
    print(f"\nتوزيع مستويات الخطر:\n{df['risk_level'].value_counts().to_string()}")
    print(f"\nإحصائيات TWSA:\n{df['twsa_mm'].describe().round(2).to_string()}")
