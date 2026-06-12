"""
lstm_model.py
-------------
LSTM مبني من الصفر بـ NumPy خالص — بدون PyTorch أو TensorFlow
يتوقع منسوب المياه الجوفية (TWSA) 6 أشهر قدام

معمارية:
  input  → LSTM(hidden=64) → Dropout(0.2) → Dense(32) → Dense(horizon)

الـ LSTM يحتوي على 4 gates:
  f = sigmoid(Wf·[h,x] + bf)   ← forget gate
  i = sigmoid(Wi·[h,x] + bi)   ← input gate
  g = tanh   (Wg·[h,x] + bg)   ← cell gate
  o = sigmoid(Wo·[h,x] + bo)   ← output gate
  c = f⊙c + i⊙g
  h = o⊙tanh(c)
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


import sys, logging, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, MODELS_DIR, BASE_DIR

log = logging.getLogger(__name__)
REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# دوال التفعيل
# ════════════════════════════════════════════════════════════
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(s):   return s * (1.0 - s)
def tanh_grad(t):      return 1.0 - t**2


# ════════════════════════════════════════════════════════════
# خلية LSTM واحدة
# ════════════════════════════════════════════════════════════
class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng  = np.random.default_rng(seed)
        d    = input_size + hidden_size
        # Xavier initialisation
        k    = np.sqrt(1.0 / hidden_size)

        def W(rows): return rng.uniform(-k, k, (rows, d)).astype(np.float64)
        def b(rows): return np.zeros(rows, np.float64)

        # Concatenated weight matrices [f, i, g, o] per layer
        self.W  = np.vstack([W(hidden_size), W(hidden_size),
                              W(hidden_size), W(hidden_size)])   # (4H, d)
        self.b  = np.concatenate([b(hidden_size)]*4)             # (4H,)

        self.hidden_size = hidden_size
        self.input_size  = input_size

    def forward(self, x, h_prev, c_prev):
        """x:(input,) → h:(hidden,), c:(hidden,)"""
        H  = self.hidden_size
        xh = np.concatenate([x, h_prev])        # (d,)
        z  = self.W @ xh + self.b               # (4H,)

        f  = sigmoid(z[0*H:1*H])
        i  = sigmoid(z[1*H:2*H])
        g  = np.tanh(z[2*H:3*H])
        o  = sigmoid(z[3*H:4*H])

        c  = f * c_prev + i * g
        h  = o * np.tanh(c)
        return h, c, (f, i, g, o, c, h, xh, c_prev)

    @property
    def params(self):  return [self.W, self.b]

    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db


# ════════════════════════════════════════════════════════════
# طبقة Dense
# ════════════════════════════════════════════════════════════
class Dense:
    def __init__(self, in_sz, out_sz, activation="linear", seed=1):
        rng   = np.random.default_rng(seed)
        k     = np.sqrt(2.0 / in_sz)
        self.W = rng.normal(0, k, (out_sz, in_sz)).astype(np.float64)
        self.b = np.zeros(out_sz, np.float64)
        self.activation = activation

    def forward(self, x):
        z = self.W @ x + self.b
        if self.activation == "relu":
            return np.maximum(0, z), z
        return z, z

    @property
    def params(self):  return [self.W, self.b]

    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db


# ════════════════════════════════════════════════════════════
# الشبكة الكاملة  LSTM → Dense → Dense
# ════════════════════════════════════════════════════════════
class WaterLSTM:
    """
    يأخذ تسلسل طوله seq_len → يتوقع horizon خطوة قادمة
    """
    def __init__(self, input_size=6, hidden_size=64, horizon=6, seq_len=12, seed=42):
        self.lstm    = LSTMCell(input_size, hidden_size, seed)
        self.dense1  = Dense(hidden_size, 32, activation="relu", seed=seed+1)
        self.dense2  = Dense(32, horizon, activation="linear",  seed=seed+2)
        self.hidden_size = hidden_size
        self.horizon     = horizon
        self.seq_len     = seq_len
        self.input_size  = input_size
        self.train_losses = []
        self.val_losses   = []

    def forward(self, X):
        """X: (seq_len, input_size) → pred: (horizon,)"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        cache = []
        for t in range(X.shape[0]):
            h, c, step_cache = self.lstm.forward(X[t], h, c)
            cache.append(step_cache)
        # Dense layers
        a1, z1 = self.dense1.forward(h)
        pred, _ = self.dense2.forward(a1)
        return pred, (cache, h, a1, z1)

    def _bptt(self, X, pred, target, cache_full):
        """Backprop through time — returns gradients"""
        cache, h_last, a1, z1 = cache_full
        H = self.hidden_size

        # Loss: MSE
        d_pred = 2.0 * (pred - target) / len(target)   # (horizon,)

        # Dense2 grads
        dW2  = np.outer(d_pred, a1)
        db2  = d_pred.copy()
        dA1  = self.dense2.W.T @ d_pred

        # Dense1 grads (ReLU)
        dZ1  = dA1 * (z1 > 0).astype(float)
        dW1  = np.outer(dZ1, h_last)
        db1  = dZ1.copy()
        dh   = self.dense1.W.T @ dZ1

        # BPTT through LSTM sequence
        dc   = np.zeros(H)
        dW_lstm = np.zeros_like(self.lstm.W)
        db_lstm = np.zeros_like(self.lstm.b)

        for t in reversed(range(len(cache))):
            f, i, g, o, c_t, h_t, xh, c_prev = cache[t]
            tc = np.tanh(c_t)

            # gate: output
            do   = dh * tc
            ddo  = do * sigmoid_grad(o)

            # cell state
            dc   = dc + dh * o * tanh_grad(tc)
            df   = dc * c_prev
            di   = dc * g
            dg   = dc * i

            ddf  = df * sigmoid_grad(f)
            ddi  = di * sigmoid_grad(i)
            ddg  = dg * tanh_grad(g)

            dz   = np.concatenate([ddf, ddi, ddg, ddo])   # (4H,)
            dW_lstm += np.outer(dz, xh)
            db_lstm += dz

            d_xh = self.lstm.W.T @ dz
            dh   = d_xh[self.input_size:]           # hidden part
            dc   = dc * f

        # Gradient clipping
        for g_arr in [dW_lstm, db_lstm, dW1, db1, dW2, db2]:
            np.clip(g_arr, -1.0, 1.0, out=g_arr)

        return dW_lstm, db_lstm, dW1, db1, dW2, db2

    def train(self, X_train, y_train, X_val, y_val,
              epochs=80, lr=0.002, batch_size=16, lr_decay=0.97):

        n = len(X_train)
        best_val_loss = np.inf
        best_W  = [p.copy() for layer in [self.lstm, self.dense1, self.dense2]
                   for p in layer.params]
        patience, wait = 12, 0

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                batch = idx[start:start+batch_size]
                dW_lstm_acc = np.zeros_like(self.lstm.W)
                db_lstm_acc = np.zeros_like(self.lstm.b)
                dW1_acc = np.zeros_like(self.dense1.W)
                db1_acc = np.zeros_like(self.dense1.b)
                dW2_acc = np.zeros_like(self.dense2.W)
                db2_acc = np.zeros_like(self.dense2.b)
                batch_loss = 0.0

                for j in batch:
                    pred, cache_full = self.forward(X_train[j])
                    loss = np.mean((pred - y_train[j])**2)
                    batch_loss += loss
                    grads = self._bptt(X_train[j], pred, y_train[j], cache_full)
                    dW_lstm_acc += grads[0]; db_lstm_acc += grads[1]
                    dW1_acc     += grads[2]; db1_acc     += grads[3]
                    dW2_acc     += grads[4]; db2_acc     += grads[5]

                sz = len(batch)
                lr_eff = lr * (lr_decay ** epoch)
                self.lstm.update(dW_lstm_acc/sz, db_lstm_acc/sz, lr_eff)
                self.dense1.update(dW1_acc/sz, db1_acc/sz, lr_eff)
                self.dense2.update(dW2_acc/sz, db2_acc/sz, lr_eff)
                epoch_loss += batch_loss

            epoch_loss /= n

            # Validation
            val_preds = np.array([self.forward(X_val[j])[0] for j in range(len(X_val))])
            val_loss  = np.mean((val_preds - y_val)**2)

            self.train_losses.append(epoch_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = [p.copy() for layer in [self.lstm, self.dense1, self.dense2]
                          for p in layer.params]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    log.info(f"  Early stop at epoch {epoch+1} | best val_loss={best_val_loss:.4f}")
                    break

            if (epoch+1) % 10 == 0:
                log.info(f"  Epoch {epoch+1:3d} | train={epoch_loss:.4f} | val={val_loss:.4f} | lr={lr_eff:.5f}")

        # Restore best weights
        params = best_W
        self.lstm.W[:]   = params[0]; self.lstm.b[:]   = params[1]
        self.dense1.W[:] = params[2]; self.dense1.b[:] = params[3]
        self.dense2.W[:] = params[4]; self.dense2.b[:] = params[5]
        return best_val_loss

    def predict(self, X):
        return np.array([self.forward(X[i])[0] for i in range(len(X))])

    def save(self, path):
        np.savez(path,
            lstm_W=self.lstm.W, lstm_b=self.lstm.b,
            d1_W=self.dense1.W, d1_b=self.dense1.b,
            d2_W=self.dense2.W, d2_b=self.dense2.b,
            meta=np.array([self.input_size, self.hidden_size,
                           self.horizon, self.seq_len]))
        log.info(f"Model saved: {path}.npz")

    @classmethod
    def load(cls, path):
        d = np.load(str(path)+".npz")
        meta = d["meta"].astype(int)
        m = cls(input_size=meta[0], hidden_size=meta[1],
                horizon=meta[2], seq_len=meta[3])
        m.lstm.W[:]   = d["lstm_W"]; m.lstm.b[:]   = d["lstm_b"]
        m.dense1.W[:] = d["d1_W"];  m.dense1.b[:] = d["d1_b"]
        m.dense2.W[:] = d["d2_W"];  m.dense2.b[:] = d["d2_b"]
        return m


# ════════════════════════════════════════════════════════════
# Feature Engineering
# ════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # لاغات زمنية
    for lag in [1, 2, 3, 6, 12]:
        df[f"twsa_lag{lag}"] = df["twsa_mm"].shift(lag)

    # متوسطات متحركة
    for w in [3, 6, 12]:
        df[f"twsa_ma{w}"] = df["twsa_mm"].rolling(w).mean()

    # معدل التغيير
    df["twsa_diff1"]  = df["twsa_mm"].diff(1)
    df["twsa_diff12"] = df["twsa_mm"].diff(12)

    # دورة موسمية (sin/cos encoding)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # سنة مُطبَّعة
    year_min = df["date"].dt.year.min()
    df["year_norm"] = (df["date"].dt.year - year_min) / 22.0

    df = df.dropna().reset_index(drop=True)
    return df


def make_sequences(df, feature_cols, target_col, seq_len=12, horizon=6):
    X, y = [], []
    vals = df[feature_cols].values.astype(np.float64)
    tgt  = df[target_col].values.astype(np.float64)

    for i in range(len(df) - seq_len - horizon + 1):
        X.append(vals[i:i+seq_len])
        y.append(tgt[i+seq_len:i+seq_len+horizon])

    return np.array(X), np.array(y)


# ════════════════════════════════════════════════════════════
# Main training pipeline
# ════════════════════════════════════════════════════════════
def train_lstm():
    df_raw = pd.read_csv(PROCESSED_DIR / "water_data_full.csv", parse_dates=["date"])
    df     = build_features(df_raw)

    FEATURES = [
        "twsa_lag1","twsa_lag2","twsa_lag3","twsa_lag6","twsa_lag12",
        "twsa_ma3","twsa_ma6","twsa_ma12",
        "twsa_diff1","twsa_diff12",
        "precipitation_mm","agri_demand_mm",
        "month_sin","month_cos","year_norm"
    ]
    FEATURES = [f for f in FEATURES if f in df.columns]

    # تطبيع
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    df[FEATURES] = scaler_X.fit_transform(df[FEATURES])
    df["twsa_scaled"] = scaler_y.fit_transform(df[["twsa_mm"]])

    SEQ_LEN, HORIZON = 12, 6
    X, y = make_sequences(df, FEATURES, "twsa_scaled", SEQ_LEN, HORIZON)

    # تقسيم زمني (لا نخلط الماضي بالمستقبل)
    split = int(len(X) * 0.80)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    log.info(f"Training samples: {len(X_tr)} | Val: {len(X_val)} | Features: {len(FEATURES)}")

    model = WaterLSTM(input_size=len(FEATURES), hidden_size=64,
                      horizon=HORIZON, seq_len=SEQ_LEN)

    t0 = time.perf_counter()
    best_val = model.train(X_tr, y_tr, X_val, y_val, epochs=80, lr=0.003)
    elapsed  = time.perf_counter() - t0

    log.info(f"Training done in {elapsed:.1f}s | best val_loss={best_val:.4f}")

    # Predictions on validation set
    preds_scaled = model.predict(X_val)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).reshape(preds_scaled.shape)
    truth = scaler_y.inverse_transform(y_val.reshape(-1,1)).reshape(y_val.shape)

    # Metrics per horizon step
    mae_per_step  = np.abs(preds - truth).mean(axis=0)
    rmse_per_step = np.sqrt(((preds - truth)**2).mean(axis=0))
    overall_mae   = mae_per_step.mean()
    overall_rmse  = rmse_per_step.mean()

    model.save(str(MODELS_DIR / "water_lstm"))

    return model, preds, truth, mae_per_step, rmse_per_step, {
        "overall_mae": overall_mae, "overall_rmse": overall_rmse,
        "elapsed_s": elapsed, "n_features": len(FEATURES),
        "n_train": len(X_tr), "n_val": len(X_val),
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "features": FEATURES,
    }
