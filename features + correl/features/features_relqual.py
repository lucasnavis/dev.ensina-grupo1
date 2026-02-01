import os
import numpy as np
import pandas as pd

# ====== CONFIG ======
INPUT_CSV = "data/crypto_5y_ohlcv.csv"
OUTPUT_CSV = "features/features_relqual.csv"

# Benchmark (use "BTC" ou "ETH" conforme seu dataset)
BENCH_TICKER = "BTC"

# Janelas
W_BETA = 60
W_RS = 30
W_IDIO = 30
W_IR = 30


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # padroniza nomes comuns
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "ticker" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
    if "symbol" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"symbol": "Ticker"})
    if "close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"close": "Close"})
    return df


def rolling_compound_return(ret: pd.Series, window: int) -> pd.Series:
    # (1+r)^window - 1
    return (1 + ret).rolling(window).apply(np.prod, raw=True) - 1


def rolling_beta(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    cov_xy = x.rolling(window).cov(y)
    var_y = y.rolling(window).var()
    return cov_xy / var_y


def main():
    df = pd.read_csv(INPUT_CSV)
    df = _normalize_columns(df)

    required = {"Date", "Ticker", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltando colunas no OHLCV: {missing}. Colunas atuais: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # retornos diários por ativo
    df["ret_1d"] = df.groupby("Ticker")["Close"].pct_change()

    # ====== BENCHMARK: série do BTC (ou ETH) ======
    bench = df[df["Ticker"] == BENCH_TICKER][["Date", "ret_1d"]].copy()
    if bench.empty:
        raise ValueError(
            f"Não encontrei BENCH_TICKER='{BENCH_TICKER}' no dataset. "
            f"Tickers disponíveis (exemplo): {df['Ticker'].unique()[:20]}"
        )

    # garante 1 linha por Date
    bench = bench.groupby("Date", as_index=False)["ret_1d"].mean()
    bench = bench.rename(columns={"ret_1d": "mkt_ret_1d"})

    # junta benchmark em todos os ativos
    df = df.merge(bench, on="Date", how="left")

    # ====== FEATURES RELATIVAS ======
    # RS 30d: retorno composto do ativo - retorno composto do benchmark
    df["asset_30d"] = df.groupby("Ticker")["ret_1d"].transform(lambda s: rolling_compound_return(s, W_RS))
    bench_30d = bench.copy()
    bench_30d["mkt_30d"] = rolling_compound_return(bench_30d["mkt_ret_1d"], W_RS)
    df = df.merge(bench_30d[["Date", "mkt_30d"]], on="Date", how="left")
    df["rs_30d"] = df["asset_30d"] - df["mkt_30d"]

    # Beta e correlação 60d vs benchmark
    df["beta_60d"] = df.groupby("Ticker").apply(
        lambda g: rolling_beta(g["ret_1d"], g["mkt_ret_1d"], W_BETA)
    ).reset_index(level=0, drop=True)

    df["corr_60d"] = df.groupby("Ticker").apply(
        lambda g: g["ret_1d"].rolling(W_BETA).corr(g["mkt_ret_1d"])
    ).reset_index(level=0, drop=True)

    # Residual e risco idiossincrático
    df["resid"] = df["ret_1d"] - df["beta_60d"] * df["mkt_ret_1d"]
    df["idio_vol_30d"] = df.groupby("Ticker")["resid"].transform(lambda s: s.rolling(W_IDIO).std())

    # IR 30d (Information Ratio) usando excesso vs benchmark
    df["excess_1d"] = df["ret_1d"] - df["mkt_ret_1d"]
    ex_mean = df.groupby("Ticker")["excess_1d"].transform(lambda s: s.rolling(W_IR).mean())
    ex_std = df.groupby("Ticker")["excess_1d"].transform(lambda s: s.rolling(W_IR).std())
    df["ir_30d"] = ex_mean / ex_std

    out = df[["Date", "Ticker", "rs_30d", "beta_60d", "corr_60d", "idio_vol_30d", "ir_30d"]].copy()
    # Only drop rows where ALL feature columns are NaN, not individual NaN values
    out = out.dropna(subset=["rs_30d", "beta_60d", "corr_60d", "idio_vol_30d", "ir_30d"], how="all").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"OK! Gerou: {OUTPUT_CSV}")
    print("Shape:", out.shape)
    print(out.head())
    print("Benchmark:", BENCH_TICKER)


if __name__ == "__main__":
    main()
