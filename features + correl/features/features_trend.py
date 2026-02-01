import numpy as np
import pandas as pd


# -----------------------------
# Auxiliar: slope do log-preço
# -----------------------------
def rolling_slope_logprice(close: pd.Series, window: int = 30) -> pd.Series:
    """
    Calcula a inclinação (slope) do log(preço) numa janela móvel.
    Retorna uma série com mesmo tamanho, com NaN nas primeiras janelas.
    """
    logp = np.log(close.astype(float))

    def _slope(y):
        x = np.arange(len(y))
        # polyfit retorna [slope, intercept]
        slope, _ = np.polyfit(x, y, 1)
        return slope

    return logp.rolling(window).apply(_slope, raw=True)


# -----------------------------
# Feature builder (por ticker)
# -----------------------------
def compute_trend_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()

    # Ordena por data
    g = g.sort_values("Date")

    # Garante a coluna Ticker (vem do nome do grupo)
    g["Ticker"] = group.name

    # Momentum
    g["mom_7d"] = g["Close"].pct_change(7)
    g["mom_30d"] = g["Close"].pct_change(30)

    # Regime (EMA 12-26)
    ema_12 = g["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = g["Close"].ewm(span=26, adjust=False).mean()
    g["ema_diff"] = (ema_12 - ema_26) / ema_26

    # Força/consistência (slope 30d do log-preço)
    g["slope_30d"] = rolling_slope_logprice(g["Close"], window=30)

    return g[["Date", "Ticker", "mom_7d", "mom_30d", "ema_diff", "slope_30d"]].copy()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 1) Lê o CSV de entrada
    df = pd.read_csv("data/crypto_5y_ohlcv.csv")

    # 2) Garante tipo de data e ordenação
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # 3) Calcula trend features por ticker
    df_trend = df.groupby("Ticker", group_keys=False).apply(compute_trend_features)

    # 4) Limpeza: remove linhas iniciais com NaN (por causa das janelas)
    df_trend = df_trend.dropna().reset_index(drop=True)

    # 5) Salva na pasta features/
    df_trend.to_csv("features/features_trend.csv", index=False)

    print("OK! features/features_trend.csv gerado.")
    print(df_trend.head())
    print("Tickers:", df_trend["Ticker"].unique())
