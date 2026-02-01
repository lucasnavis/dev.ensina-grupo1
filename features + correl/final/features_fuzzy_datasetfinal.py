# final/features_fuzzy_datasetfinal.py
from pathlib import Path
import numpy as np
import pandas as pd

IN_CSV = Path("final/features_normal_datasetfinal.csv")
OUT_CSV = Path("final/features_fuzzy_datasetfinal.csv")

BASE_COLS = ["Date", "Ticker"]

# =========================================================
# Membership functions
# =========================================================
def gaussian(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def trapezoidal(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    mu[(x >= b) & (x <= c)] = 1.0
    if b != a:
        m = (x > a) & (x < b)
        mu[m] = (x[m] - a) / (b - a)
    if d != c:
        m = (x > c) & (x < d)
        mu[m] = (d - x[m]) / (d - c)

    return np.clip(mu, 0.0, 1.0)

def triangular(x, a, b, c):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    if b != a:
        up = (a < x) & (x <= b)
        mu[up] = (x[up] - a) / (b - a)
    if c != b:
        down = (b < x) & (x < c)
        mu[down] = (c - x[down]) / (c - b)

    mu[x == b] = 1.0
    return np.clip(mu, 0.0, 1.0)

# =========================================================
# Helpers (thresholds via quantis -> evita "chute" manual)
# =========================================================
def qvals(s: pd.Series, qs=(0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 10:
        raise ValueError(f"Poucos dados válidos para fuzzificar: {s.name}")
    return x.quantile(list(qs)).to_dict()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

# =========================================================
# Fuzzificação por feature (1 por “tópico”, cada uma do seu jeito)
# =========================================================
def fuzz_mom_7d(mom: pd.Series):
    # tendência: queremos “bear / neutro / bull”, com zona neutra mais “plana”
    q = qvals(mom)
    q10, q25, q40, q50, q60, q75, q90 = [safe_float(q[k]) for k in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)]
    x = pd.to_numeric(mom, errors="coerce").fillna(q50).to_numpy(float)

    # neutro: trapezoidal (zona estável perto do centro)
    # bear/bull: triangular (transição mais “decisória”)
    return {
        "trend_bear": triangular(x, q10, q25, q50),
        "trend_neutral": trapezoidal(x, q40, q50, q50, q60),
        "trend_bull": triangular(x, q50, q75, q90),
    }

def fuzz_vov_60d(vov: pd.Series):
    # volatilidade (vol-of-vol): suave/ruidosa -> gaussianas
    q = qvals(vov)
    q25, q50, q75 = [safe_float(q[k]) for k in (0.25, 0.50, 0.75)]
    x = pd.to_numeric(vov, errors="coerce").fillna(q50).to_numpy(float)

    # sigma baseado em IQR (robusto)
    sigma = max((q75 - q25) / 1.349, 1e-6)  # ~std se normal
    return {
        "vol_low": gaussian(x, mu=q25, sigma=sigma),
        "vol_mid": gaussian(x, mu=q50, sigma=sigma),
        "vol_high": gaussian(x, mu=q75, sigma=sigma),
    }

def fuzz_dd_dur_60d(dd_dur: pd.Series):
    # stress: duração de drawdown costuma ser assimétrica/cauda longa -> log1p antes
    s = pd.to_numeric(dd_dur, errors="coerce")
    s = s.clip(lower=0)
    z = np.log1p(s)

    q = qvals(pd.Series(z, name="dd_dur_60d_log"))
    q10, q25, q50, q75, q90 = [safe_float(q[k]) for k in (0.10, 0.25, 0.50, 0.75, 0.90)]
    x = pd.Series(z).fillna(q50).to_numpy(float)

    return {
        "stress_low": triangular(x, q10, q25, q50),
        "stress_mid": trapezoidal(x, q25, q50, q50, q75),
        "stress_high": triangular(x, q50, q75, q90),
    }

def fuzz_autocorr_30d(ac: pd.Series):
    # qualidade/estrutura: autocorr está em [-1,1] e “neutro” é perto de 0
    x_raw = pd.to_numeric(ac, errors="coerce")
    x_raw = x_raw.clip(-1, 1)
    q = qvals(x_raw)

    q10, q30, q50, q70, q90 = [safe_float(q[k]) for k in (0.10, 0.40, 0.50, 0.60, 0.90)]
    x = x_raw.fillna(q50).to_numpy(float)

    return {
        "ac_negative": triangular(x, -1.0, q10, 0.0),
        "ac_neutral": trapezoidal(x, q30, 0.0, 0.0, q70),
        "ac_positive": triangular(x, 0.0, q90, 1.0),
    }

def fuzz_fgi_vol_30d(fgi_vol: pd.Series):
    # macro: volatilidade do FGI (proxy de “regime”/instabilidade) -> suave -> gaussianas
    q = qvals(fgi_vol)
    q20, q50, q80 = [safe_float(q[k]) for k in (0.25, 0.50, 0.75)]
    x = pd.to_numeric(fgi_vol, errors="coerce").fillna(q50).to_numpy(float)

    sigma = max((q80 - q20) / 1.349, 1e-6)
    return {
        "macro_calm": gaussian(x, mu=q20, sigma=sigma),
        "macro_neutral": gaussian(x, mu=q50, sigma=sigma),
        "macro_stress": gaussian(x, mu=q80, sigma=sigma),
    }

# Mapeia automaticamente o que existir no seu CSV
FUZZIFIERS = {
    "mom_7d": (fuzz_mom_7d, "trend"),
    "vov_60d": (fuzz_vov_60d, "vol"),
    "dd_dur_60d": (fuzz_dd_dur_60d, "stress"),
    "autocorr_30d": (fuzz_autocorr_30d, "quality"),
    "fgi_vol_30d": (fuzz_fgi_vol_30d, "macro"),
}

def main():
    print("=== RUNNING features_fuzzy_datasetfinal.py ===")
    print("Input:", IN_CSV, "| exists:", IN_CSV.exists())
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Não achei o input: {IN_CSV.resolve()}")

    df = pd.read_csv(IN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # checa base
    missing_base = [c for c in BASE_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(f"Faltando colunas base: {missing_base}. Colunas: {list(df.columns)}")

    out = df[BASE_COLS].copy()

    applied = []
    for col, (fn, group) in FUZZIFIERS.items():
        if col in df.columns:
            d = fn(df[col])
            for k, v in d.items():
                out[k] = v
            applied.append((col, group, list(d.keys())))

    if not applied:
        raise ValueError(
            "Nenhuma feature conhecida encontrada para fuzzificar. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("✅ OK! Salvo:", OUT_CSV.resolve())
    print("Shape:", out.shape)
    print("Fuzzifiers aplicados:")
    for col, group, keys in applied:
        print(f"- {group}: {col} -> {keys}")

if __name__ == "__main__":
    main()
