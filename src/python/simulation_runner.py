"""
simulation_runner.py
--------------------
يربط محرك C++ بالبيانات الحقيقية ويشغّل سيناريوهات متعددة
ويقيس الأداء مقارنةً بالـ Python الخالص
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

import sys, time, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import groundwater_engine as gwe
from config import PROCESSED_DIR, BASE_DIR, AQUIFER_PARAMS, REGION

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Python-only FD (للمقارنة)
# ─────────────────────────────────────────────────────────────
def simulate_python(h0, pump, K, S, b, dx, dy, n_steps, dt, recharge):
    nx, ny = h0.shape
    T  = K * b
    Tx = T / dx**2
    Ty = T / dy**2
    h  = h0.copy()
    means = []
    for _ in range(n_steps):
        h_new = h.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                lap = Tx*(h[i+1,j]-2*h[i,j]+h[i-1,j]) + Ty*(h[i,j+1]-2*h[i,j]+h[i,j-1])
                h_new[i,j] = h[i,j] + (dt/S)*(lap + recharge - pump[i,j])
        h = h_new
        means.append(h.mean())
    return h, means


# ─────────────────────────────────────────────────────────────
# بناء شبكة واقعية من بيانات GRACE
# ─────────────────────────────────────────────────────────────
def build_grid_from_data(nx=20, ny=26):
    grid_df = pd.read_csv(PROCESSED_DIR / "spatial_grid.csv")

    # تحويل TWSA → hydraulic head (offset واقعي)
    base_head = 120.0   # م — متوسط عمق المياه الجوفية في حوض النوبة
    h0 = np.zeros((nx, ny))

    lats = np.linspace(REGION["lat_min"], REGION["lat_max"], nx)
    lons = np.linspace(REGION["lon_min"], REGION["lon_max"], ny)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # أقرب نقطة في الـ grid
            dist = ((grid_df.latitude - lat)**2 + (grid_df.longitude - lon)**2)
            nearest = grid_df.iloc[dist.idxmin()]
            # TWSA negative = انخفاض عن المتوسط
            h0[i, j] = base_head + nearest["twsa_mean"] / 1000.0
    return h0


# ─────────────────────────────────────────────────────────────
# سيناريوهات المحاكاة
# ─────────────────────────────────────────────────────────────
# Designed so scenarios show contrasting outcomes:
# baseline   → slight depletion (pump ≈ recharge)
# high_demand → clear depletion (pump >> recharge)
# conservation → recovery (pump << recharge)
SCENARIOS = {
    "baseline":    {"pump_rate": 1.50e-5, "recharge": 1.37e-5, "label": "Baseline (current trends)"},
    "high_demand": {"pump_rate": 2.74e-5, "recharge": 1.37e-5, "label": "High demand (+100%)"},
    "conservation":{"pump_rate": 0.50e-6, "recharge": 2.74e-5, "label": "Conservation + recharge"},
}

def run_scenarios(h0, nx, ny, n_years=20):
    dx = AQUIFER_PARAMS["grid_spacing_km"] * 1000
    K  = AQUIFER_PARAMS["hydraulic_conductivity"]
    S  = AQUIFER_PARAMS["storativity"]
    b  = 50.0
    dt = 30.0
    n_steps = n_years * 12

    results = {}
    for name, cfg in SCENARIOS.items():
        pump = np.full((nx, ny), cfg["pump_rate"])
        t0   = time.perf_counter()
        res  = gwe.simulate_groundwater(
            h0, pump, K=K, S=S, b=b,
            dx=dx, dy=dx, n_steps=n_steps,
            dt=dt, recharge=cfg["recharge"]
        )
        elapsed = time.perf_counter() - t0
        results[name] = {**res, "cfg": cfg, "elapsed_s": elapsed}
        log.info(f"  {name:12s}: {elapsed*1000:.1f} ms | "
                 f"final head {np.array(res['mean_head_series'])[-1]:.2f} m")
    return results, n_steps


# ─────────────────────────────────────────────────────────────
# مقارنة الأداء C++ vs Python
# ─────────────────────────────────────────────────────────────
def benchmark(h0, nx, ny):
    dx   = AQUIFER_PARAMS["grid_spacing_km"] * 1000
    K, S, b, dt = 5.0, 0.001, 50.0, 30.0
    pump = np.full((nx, ny), 1.39e-6)
    n    = 24   # سنتان فقط للمقارنة

    log.info("Benchmarking C++ vs Python...")

    t0 = time.perf_counter()
    r_cpp = gwe.simulate_groundwater(h0, pump, K=K, S=S, b=b,
        dx=dx, dy=dx, n_steps=n, dt=dt, recharge=1.37e-5)
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_py, _ = simulate_python(h0, pump, K, S, b, dx, dx, n, dt, 1.37e-5)
    t_py = time.perf_counter() - t0

    speedup = t_py / t_cpp
    log.info(f"  C++    : {t_cpp*1000:.1f} ms")
    log.info(f"  Python : {t_py*1000:.1f} ms")
    log.info(f"  Speedup: {speedup:.1f}x")
    return {"cpp_ms": t_cpp*1000, "py_ms": t_py*1000, "speedup": speedup}


# ─────────────────────────────────────────────────────────────
# رسم النتائج
# ─────────────────────────────────────────────────────────────
def plot_simulation_results(results, n_steps, bench):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    colors = {"baseline": "#1D6FA5", "high_demand": "#E24B4A", "conservation": "#1D9E75"}
    years  = np.arange(n_steps) / 12.0

    # ── 1. head timeseries per scenario ──
    ax = axes[0, 0]
    for name, res in results.items():
        series = np.array(res["mean_head_series"])
        ax.plot(years, series, color=colors[name], lw=2, label=res["cfg"]["label"])
    ax.set_title("Mean hydraulic head over 20 years")
    ax.set_xlabel("Years"); ax.set_ylabel("Head (m)")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    # ── 2. storage change ──
    ax = axes[0, 1]
    for name, res in results.items():
        sc = np.array(res["storage_change_m3"]) / 1e9  # km³
        ax.plot(years, sc, color=colors[name], lw=2)
    ax.set_title("Cumulative storage change")
    ax.set_xlabel("Years"); ax.set_ylabel("ΔStorage (km³)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.grid(alpha=0.2)

    # ── 3. final head map (baseline) ──
    ax = axes[1, 0]
    final_h = np.array(results["baseline"]["final_head"])
    im = ax.imshow(final_h, cmap="RdYlBu", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="Head (m)")
    ax.set_title("Final hydraulic head map — baseline")
    ax.set_xlabel("Longitude grid"); ax.set_ylabel("Latitude grid")

    # ── 4. benchmark ──
    ax = axes[1, 1]
    bars = ax.bar(["Python\n(pure)", "C++\nengine"],
                  [bench["py_ms"], bench["cpp_ms"]],
                  color=["#888780", "#1D6FA5"], width=0.5)
    ax.bar_label(bars, fmt="%.1f ms", padding=3, fontsize=10)
    ax.set_title(f"Performance: C++ is {bench['speedup']:.0f}× faster")
    ax.set_ylabel("Time (ms) — 24-step simulation")
    ax.set_ylim(0, bench["py_ms"] * 1.3)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Groundwater Simulation Results — Egypt & North Africa", fontsize=13)
    plt.tight_layout()

    out = REPORTS / "simulation_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nx, ny = 20, 26
    log.info(f"Building grid {nx}x{ny} from GRACE data...")
    h0 = build_grid_from_data(nx, ny)
    log.info(f"  head range: {h0.min():.2f} – {h0.max():.2f} m")

    log.info("Running scenarios...")
    results, n_steps = run_scenarios(h0, nx, ny, n_years=20)

    bench = benchmark(h0, nx, ny)

    out = plot_simulation_results(results, n_steps, bench)
    log.info(f"Chart saved: {out}")

    print(f"\n{'='*50}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*50}")
    for name, res in results.items():
        s = np.array(res['mean_head_series'])
        change = s[-1] - s[0]   # positive = rise, negative = decline
        sc     = np.array(res['storage_change_m3'])[-1] / 1e9
        label  = "RISE" if change >= 0 else "DROP"
        print(f"  {res['cfg']['label']}")
        print(f"    Head change: {change:+.2f} m over 20 years ({label})")
        print(f"    Storage Δ  : {sc:+.3f} km³")
        print(f"    Sim time   : {res['elapsed_s']*1000:.1f} ms")
        print()
    print(f"  C++ speedup: {bench['speedup']:.1f}x over pure Python")
    print(f"{'='*50}")
