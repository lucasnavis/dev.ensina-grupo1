# features/features_extra.py
from pathlib import Path
import numpy as np
import pandas as pd

IN_OHLCV = Path("data/crypto_5y_ohlcv.csv")
IN_FGI   = Path("data/fgi.csv")
OUT_CSV  = Path("features/features_extra.csv")


# ---------- helpers ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=win).mean()
    s = x.rolling(win, min_periods=win).std()
    return (x - m) / (s.replace(0, np.nan))


def rolling_cvar(ret: pd.Series, win: int = 60, alpha: float = 0.05) -> pd.Series:
    def _cvar(a):
        a = np.asarray(a, dtype=float)
        q = np.nanquantile(a, alpha)
        tail = a[a <= q]
        return np.nanmean(tail) if tail.size else np.nan
    return ret.rolling(win, min_periods=win).apply(_cvar, raw=False)


def parkinson_vol(high: pd.Series, low: pd.Series, win: int = 30) -> pd.Series:
    hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    var = (hl ** 2).rolling(win, min_periods=win).mean() / (4 * np.log(2))
    return np.sqrt(var)


def rolling_percentile_last(x: pd.Series) -> float:
    # percentil do último valor dentro da janela (0..1)
    s = pd.Series(x)
    return float(s.rank(pct=True).iloc[-1])


def rolling_autocorr_lag1(ret: pd.Series, win: int = 30) -> pd.Series:
    # autocorr de lag 1 numa janela
    def _ac(a):
        s = pd.Series(a).dropna()
        if len(s) < 5:
            return np.nan
        return s.autocorr(lag=1)
    return ret.rolling(win, min_periods=win).apply(_ac, raw=False)


def main():
    if not IN_OHLCV.exists():
        raise FileNotFoundError(f"Não achei {IN_OHLCV.resolve()}")
    if not IN_FGI.exists():
        raise FileNotFoundError(f"Não achei {IN_FGI.resolve()}")

    ohlcv = pd.read_csv(IN_OHLCV)

    # checks mínimos
    for c in ["Date", "Ticker", "Close", "High", "Low"]:
        if c not in ohlcv.columns:
            raise ValueError(f"IN_OHLCV precisa ter coluna {c}")

    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    ohlcv = ohlcv.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # returns
    ohlcv["ret_1d"] = ohlcv.groupby("Ticker")["Close"].pct_change()

    # -----------------------
    # TREND candidates
    # -----------------------
    ohlcv["mom_14d"] = ohlcv.groupby("Ticker")["Close"].pct_change(14)
    ohlcv["mom_60d"] = ohlcv.groupby("Ticker")["Close"].pct_change(60)

    ema12 = ohlcv.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = ohlcv.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    ohlcv["ema_cross_12_26"] = (ema12 / ema26) - 1.0

    ohlcv["rsi_14"] = ohlcv.groupby("Ticker")["Close"].transform(lambda s: rsi(s, 14))
    ohlcv["rsi_14_n"] = (ohlcv["rsi_14"] - 50.0) / 50.0

    # -----------------------
    # VOL candidates
    # -----------------------
    ohlcv["rv_14d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: s.rolling(14, min_periods=14).std())
    ohlcv["rv_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: s.rolling(60, min_periods=60).std())

    def downside_std(a):
        s = pd.Series(a)
        s = s[s < 0]
        return s.std() if len(s) else np.nan

    ohlcv["downside_vol_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=60).apply(lambda a: downside_std(a), raw=False)
    )

    ohlcv["parkinson_vol_30d"] = ohlcv.groupby("Ticker", group_keys=False).apply(
        lambda g: parkinson_vol(g["High"].astype(float), g["Low"].astype(float), 30)
    )

    ohlcv["vov_60d"] = ohlcv.groupby("Ticker")["rv_14d"].transform(lambda s: s.rolling(60, min_periods=60).std())

    # -----------------------
    # STRESS candidates
    # -----------------------
    ohlcv["max_loss_30d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: s.rolling(30, min_periods=30).min())
    ohlcv["cvar_5_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: rolling_cvar(s, 60, 0.05))

    # duration desde o último topo
    def dd_duration(close: pd.Series) -> pd.Series:
        roll_max = close.cummax()
        is_new_high = close == roll_max
        dur = np.zeros(len(close), dtype=float)
        last_high = -1
        for i, nh in enumerate(is_new_high.values):
            if nh:
                last_high = i
                dur[i] = 0.0
            else:
                dur[i] = float(i - last_high) if last_high >= 0 else np.nan
        return pd.Series(dur, index=close.index)

    ohlcv["dd_dur"] = ohlcv.groupby("Ticker")["Close"].transform(dd_duration)
    ohlcv["dd_dur_60d"] = ohlcv.groupby("Ticker")["dd_dur"].transform(lambda s: s.rolling(60, min_periods=60).max())

    # -----------------------
    # QUALITY candidates (novas)
    # -----------------------
    ohlcv["hit_ratio_30d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(
        lambda s: (s > 0).rolling(30, min_periods=30).mean()
    )

    # profit factor 60d: sum(gains)/abs(sum(losses))
    def profit_factor(a):
        s = pd.Series(a).dropna()
        if len(s) < 10:
            return np.nan
        gains = s[s > 0].sum()
        losses = s[s < 0].sum()
        if losses == 0:
            return np.nan
        return gains / abs(losses)

    ohlcv["profit_factor_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=60).apply(lambda a: profit_factor(a), raw=False)
    )

    # omega 60d (ganhos vs perdas em 0)
    def omega_ratio(a, thr=0.0):
        s = pd.Series(a).dropna()
        if len(s) < 10:
            return np.nan
        up = (s - thr)
        gains = up[up > 0].sum()
        losses = (-up[up < 0]).sum()
        if losses == 0:
            return np.nan
        return gains / losses

    ohlcv["omega_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=60).apply(lambda a: omega_ratio(a, 0.0), raw=False)
    )

    ohlcv["autocorr_30d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: rolling_autocorr_lag1(s, 30))
    ohlcv["skew_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: s.rolling(60, min_periods=60).skew())
    ohlcv["kurt_60d"] = ohlcv.groupby("Ticker")["ret_1d"].transform(lambda s: s.rolling(60, min_periods=60).kurt())

    # -----------------------
    # MACRO candidates (novas)
    # -----------------------
    fgi = pd.read_csv(IN_FGI)
    if "Date" not in fgi.columns:
        raise ValueError("IN_FGI precisa ter coluna Date")
    fgi["Date"] = pd.to_datetime(fgi["Date"])
    fgi = fgi.sort_values("Date").reset_index(drop=True)

    # tenta achar coluna de valor do FGI
    fgi_val_col = None
    for c in ["fgi", "FGI", "value", "Value", "index", "Index"]:
        if c in fgi.columns:
            fgi_val_col = c
            break
    if fgi_val_col is None:
        raise ValueError("Não achei coluna de valor do FGI (tente fgi/value/index)")

    fgi["fgi"] = pd.to_numeric(fgi[fgi_val_col], errors="coerce")

    fgi["fgi_chg_1d"] = fgi["fgi"].diff(1)
    fgi["fgi_chg_7d"] = fgi["fgi"].diff(7)
    fgi["fgi_chg_14d"] = fgi["fgi"].diff(14)

    fgi["fgi_ema_14"] = fgi["fgi"].ewm(span=14, adjust=False).mean()
    fgi["fgi_gap_ema14"] = fgi["fgi"] - fgi["fgi_ema_14"]

    fgi["fgi_z_90d"] = zscore(fgi["fgi"], 90)

    # vol do sentimento (std das variações 1d)
    fgi["fgi_vol_30d"] = fgi["fgi_chg_1d"].rolling(30, min_periods=30).std()

    # regime relativo (percentil 180d)
    fgi["fgi_rank_180d"] = fgi["fgi"].rolling(180, min_periods=180).apply(rolling_percentile_last, raw=False)

    fgi = fgi[[
        "Date", "fgi",
        "fgi_chg_1d", "fgi_chg_7d", "fgi_chg_14d",
        "fgi_ema_14", "fgi_gap_ema14",
        "fgi_z_90d", "fgi_vol_30d", "fgi_rank_180d"
    ]]

    # merge macro por Date (FGI é global)
    base = ohlcv.merge(fgi, on="Date", how="left")

    # ---------- output ----------
    out_cols = [
        "Date", "Ticker",
        # trend
        "mom_14d", "mom_60d", "ema_cross_12_26", "rsi_14_n",
        # vol
        "rv_14d", "rv_60d", "downside_vol_60d", "parkinson_vol_30d", "vov_60d",
        # stress
        "max_loss_30d", "cvar_5_60d", "dd_dur_60d",
        # quality (novas)
        "hit_ratio_30d", "profit_factor_60d", "omega_60d", "autocorr_30d", "skew_60d", "kurt_60d",
        # macro (novas)
        "fgi", "fgi_chg_1d", "fgi_chg_7d", "fgi_chg_14d",
        "fgi_ema_14", "fgi_gap_ema14", "fgi_z_90d", "fgi_vol_30d", "fgi_rank_180d",
    ]

    missing = [c for c in out_cols if c not in base.columns]
    if missing:
        raise ValueError(f"Colunas faltando no output: {missing}")

    out = base[out_cols].copy()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("✅ OK! Salvo:", OUT_CSV.resolve())
    print("Shape:", out.shape)
    print("Colunas:", list(out.columns))


if __name__ == "__main__":
    main()
