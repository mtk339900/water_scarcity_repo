/*
groundwater_engine.cpp
======================
محرك محاكاة المياه الجوفية باستخدام معادلة Darcy + Finite Difference Method

المعادلة الحاكمة (2D Groundwater Flow Equation):
    S * ∂h/∂t = ∇·(T ∇h) + W

حيث:
    h  = hydraulic head (ارتفاع المياه بالأمتار)
    S  = storativity (معامل التخزين، بلا وحدة)
    T  = transmissivity (m²/day) = K * b
    W  = source/sink term (إعادة شحن - ضخ، m/day)
    K  = hydraulic conductivity (m/day)

الحل العددي: Finite Difference Explicit Scheme
    h[i,j,t+1] = h[i,j,t] + (dt/S) * (
        Tx*(h[i+1,j] - 2h[i,j] + h[i-1,j]) / dx²
      + Ty*(h[i,j+1] - 2h[i,j] + h[i,j-1]) / dy²
      + W[i,j]
    )

شرط الاستقرار (Courant–Friedrichs–Lewy):
    dt ≤ S * dx² / (4 * T)
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────
// هيكل معاملات الحوض
// ─────────────────────────────────────────────────────────────
struct AquiferParams {
    double K;           // hydraulic conductivity (m/day)
    double S;           // storativity
    double b;           // aquifer thickness (m)
    double dx;          // grid spacing x (m)
    double dy;          // grid spacing y (m)
    double dt;          // time step (days)
    double recharge;    // recharge rate (m/day)
    double pumping;     // pumping rate (m/day), positive = extraction
};

// ─────────────────────────────────────────────────────────────
// حساب شرط الاستقرار CFL
// ─────────────────────────────────────────────────────────────
double compute_stable_dt(const AquiferParams& p) {
    double T  = p.K * p.b;
    double dt_stable = 0.45 * p.S * p.dx * p.dx / (2.0 * T);
    return dt_stable;
}

// ─────────────────────────────────────────────────────────────
// خطوة زمنية واحدة — Explicit Finite Difference
// ─────────────────────────────────────────────────────────────
void step_forward(
    std::vector<double>&       h,
    const std::vector<double>& pumping_grid,
    int nx, int ny,
    const AquiferParams& p)
{
    double T  = p.K * p.b;
    double Tx = T / (p.dx * p.dx);
    double Ty = T / (p.dy * p.dy);
    double dt_S = p.dt / p.S;

    std::vector<double> h_new(h);   // copy → compute into h_new

    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            int c  = i * ny + j;
            int up = (i-1)*ny + j;
            int dn = (i+1)*ny + j;
            int lt = i*ny + (j-1);
            int rt = i*ny + (j+1);

            double laplacian_x = Tx * (h[dn] - 2.0*h[c] + h[up]);
            double laplacian_y = Ty * (h[rt] - 2.0*h[c] + h[lt]);

            // W = recharge - pumping (m/day, نسبي للوحدة المساحية)
            double W = p.recharge - pumping_grid[c];

            h_new[c] = h[c] + dt_S * (laplacian_x + laplacian_y + W);
        }
    }
    h = h_new;
}

// ─────────────────────────────────────────────────────────────
// دالة المحاكاة الرئيسية
// ─────────────────────────────────────────────────────────────
py::dict simulate_groundwater(
    py::array_t<double> initial_head_arr,   // (nx, ny) مصفوفة الارتفاعات الأولية
    py::array_t<double> pumping_arr,         // (nx, ny) شبكة الضخ
    double K,           // hydraulic conductivity m/day
    double S,           // storativity
    double b,           // سماكة الحوض m
    double dx,          // تباعد الشبكة x بالأمتار
    double dy,          // تباعد الشبكة y بالأمتار
    int    n_steps,     // عدد الخطوات الزمنية
    double dt,          // الخطوة الزمنية بالأيام
    double recharge     // معدل إعادة الشحن m/day
) {
    // ── تحقق من الأبعاد ──
    auto head_buf = initial_head_arr.request();
    auto pump_buf = pumping_arr.request();

    if (head_buf.ndim != 2 || pump_buf.ndim != 2)
        throw std::invalid_argument("Arrays must be 2D");
    if (head_buf.shape[0] != pump_buf.shape[0] ||
        head_buf.shape[1] != pump_buf.shape[1])
        throw std::invalid_argument("head and pumping arrays must have same shape");

    int nx = head_buf.shape[0];
    int ny = head_buf.shape[1];

    // ── بناء هيكل المعاملات ──
    AquiferParams p{K, S, b, dx, dy, dt, recharge, 0.0};

    // ── تحقق من شرط الاستقرار ──
    double dt_max = compute_stable_dt(p);
    if (dt > dt_max * 1.05) {
        throw std::invalid_argument(
            "Time step dt=" + std::to_string(dt) +
            " exceeds stability limit " + std::to_string(dt_max) +
            ". Reduce dt or increase grid spacing."
        );
    }

    // ── نسخ البيانات إلى vectors ──
    double* head_ptr = static_cast<double*>(head_buf.ptr);
    double* pump_ptr = static_cast<double*>(pump_buf.ptr);

    std::vector<double> h(head_ptr, head_ptr + nx*ny);
    std::vector<double> pumping_flat(pump_ptr, pump_ptr + nx*ny);

    // ── تخزين نتائج كل خطوة ──
    // نحفظ: المتوسط، الحد الأدنى، الحد الأقصى لكل خطوة (توفير الذاكرة)
    std::vector<double> mean_head(n_steps);
    std::vector<double> min_head(n_steps);
    std::vector<double> max_head(n_steps);
    std::vector<double> total_storage_change(n_steps);

    double initial_mean = 0.0;
    for (double v : h) initial_mean += v;
    initial_mean /= h.size();

    // ── حلقة المحاكاة الرئيسية ──
    for (int t = 0; t < n_steps; ++t) {
        step_forward(h, pumping_flat, nx, ny, p);

        double sum = 0.0, mn = h[0], mx = h[0];
        for (double v : h) {
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        mean_head[t]           = sum / h.size();
        min_head[t]            = mn;
        max_head[t]            = mx;
        total_storage_change[t] = (mean_head[t] - initial_mean) * S * dx * dy * nx * ny;
    }

    // ── حالة الشبكة النهائية → numpy array ──
    py::array_t<double> final_head({nx, ny});
    auto out_buf = final_head.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    for (int i = 0; i < nx*ny; ++i) out_ptr[i] = h[i];

    // ── تجميع النتائج ──
    py::dict result;
    result["final_head"]           = final_head;
    result["mean_head_series"]     = mean_head;
    result["min_head_series"]      = min_head;
    result["max_head_series"]      = max_head;
    result["storage_change_m3"]    = total_storage_change;
    result["dt_used"]              = dt;
    result["dt_stability_limit"]   = dt_max;
    result["n_steps"]              = n_steps;
    result["grid_nx"]              = nx;
    result["grid_ny"]              = ny;
    return result;
}

// ─────────────────────────────────────────────────────────────
// حساب تدرج الضغط الهيدروليكي (للتصور)
// ─────────────────────────────────────────────────────────────
py::array_t<double> compute_gradient_magnitude(
    py::array_t<double> head_arr,
    double dx, double dy)
{
    auto buf = head_arr.request();
    if (buf.ndim != 2) throw std::invalid_argument("Must be 2D");
    int nx = buf.shape[0], ny = buf.shape[1];
    double* h = static_cast<double*>(buf.ptr);

    py::array_t<double> grad({nx, ny});
    auto gbuf = grad.request();
    double* g = static_cast<double*>(gbuf.ptr);

    for (int i = 1; i < nx-1; ++i) {
        for (int j = 1; j < ny-1; ++j) {
            double dhdx = (h[(i+1)*ny+j] - h[(i-1)*ny+j]) / (2*dx);
            double dhdy = (h[i*ny+(j+1)] - h[i*ny+(j-1)]) / (2*dy);
            g[i*ny+j] = std::sqrt(dhdx*dhdx + dhdy*dhdy);
        }
    }
    return grad;
}

// ─────────────────────────────────────────────────────────────
// pybind11 module
// ─────────────────────────────────────────────────────────────
PYBIND11_MODULE(groundwater_engine, m) {
    m.doc() = "Groundwater flow simulation engine (Darcy + 2D FD)";

    m.def("simulate_groundwater", &simulate_groundwater,
        py::arg("initial_head"),
        py::arg("pumping"),
        py::arg("K")        = 5.0,
        py::arg("S")        = 0.001,
        py::arg("b")        = 50.0,
        py::arg("dx")       = 10000.0,
        py::arg("dy")       = 10000.0,
        py::arg("n_steps")  = 365,
        py::arg("dt")       = 1.0,
        py::arg("recharge") = 0.0000137,
        "Run 2D groundwater flow simulation"
    );

    m.def("compute_gradient_magnitude", &compute_gradient_magnitude,
        py::arg("head"), py::arg("dx"), py::arg("dy"),
        "Compute hydraulic gradient magnitude at each grid point"
    );

    m.def("compute_stable_dt",
        [](double K, double S, double b, double dx) {
            AquiferParams p; p.K=K; p.S=S; p.b=b; p.dx=dx; p.dy=dx;
            p.dt=1.0; p.recharge=0.0; p.pumping=0.0;
            return compute_stable_dt(p);
        },
        py::arg("K"), py::arg("S"), py::arg("b"), py::arg("dx"),
        "Return maximum stable time step (days) for given parameters"
    );
}
