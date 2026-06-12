"""
config.py — Central configuration for the Water Scarcity Analysis project.
All paths are resolved relative to the project root, making the project
portable across Linux, macOS, and Windows.
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


# ── Project root: two levels up from this file (src/python/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Data directories
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRACE_DIR     = DATA_DIR / "nasa_grace"

# ── Output directories
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"

# ── Auto-create all directories on import
for _d in [RAW_DIR, PROCESSED_DIR, GRACE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Study region: Egypt and North Africa
REGION = {
    "name":    "Egypt and North Africa",
    "lat_min": 22.0,
    "lat_max": 32.0,
    "lon_min": 24.0,
    "lon_max": 37.0,
}

# ── Aquifer physical parameters (Nubian Sandstone Aquifer System)
AQUIFER_PARAMS = {
    "porosity":               0.20,   # fraction
    "hydraulic_conductivity": 5.0,    # m/day
    "storativity":            0.001,  # dimensionless
    "recharge_rate":          5.0,    # mm/year
    "grid_spacing_km":        10.0,   # km
}

# ── Time series span
TIME_SERIES = {
    "start_year": 2002,
    "end_year":   2023,
    "frequency":  "monthly",
}

# ── Risk classification thresholds (mm of water equivalent)
RISK_THRESHOLDS = {
    "low":    -50,
    "medium": -100,
    "high":   -200,
}

if __name__ == "__main__":
    print(f"Project root : {BASE_DIR}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"Region       : {REGION['name']}")
