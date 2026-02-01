import os
import numpy as np
import pandas as pd


# =========================
# 1) Configurações
# =========================
INPUT_CSV = os.path.join("data", "crypto_5y_ohlcv.csv")
OUTPUT_CSV = os.path.join("features", "features_vol.csv")


# =========================
# 2) Funções auxiliares
# =========================
def rolling_max_drawdown(close: pd.Series, window: int = 30) -> pd.Series:
    """
    Max Drawdown em janela móvel:
    - Calcula o pico acumulado dentro da janela e a queda relativa até o valor atual
    - Retorna o pior (máximo) drawdown dentro da janela, em valor positivo (ex: 0.25 = 25%)
    """
    # drawdown instantâneo vs pico passado
    roll_max = close.rolling(window=window, min_periods=window).max()
    dd = (roll_max - close) / roll_max
    # pior dd dentro da janela (máximo)
    return dd.rolling(window=window, min_periods=window).max()


def compute_vol_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de volatilidade para um único ticker (um grupo do groupby).
    Saída com colunas: Date, Ticker, ret_1d, vol_7d, vol_30d, atrp_14, max_dd_30d
    """
    g = group.copy()

    # Ordena por data (garante rolling correto)
    g = g.sort_values("Date")

    # Garante que exista a coluna Ticker (evita KeyError após apply)
    g["Ticker"] = group.name

    # =========================
    # (A) Retornos e vol realizada
    # =========================
    g["ret_1d"] = g["Close"].pct_change(1)

    # Volatilidade realizada: desvio padrão dos retornos na janela
    # ddof=0 para padronizar (pop std)
    g["vol_7d"] = g["ret_1d"].rolling(window=7, min_periods=7).std(ddof=0)
    g["vol_30d"] = g["ret_1d"].rolling(window=30, min_periods=30).std(ddof=0)

    # =========================
    # (B) ATR% (intradiário / range)
    # =========================
    prev_close = g["Close"].shift(1)
    tr1 = g["High"] - g["Low"]
    tr2 = (g["High"] - prev_close).abs()
    tr3 = (g["Low"] - prev_close).abs()

    g["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR 14 (média móvel simples do True Range)
    g["atr_14"] = g["tr"].rolling(window=14, min_periods=14).mean()

    # ATR percentual (normalizado pelo preço)
    g["atrp_14"] = g["atr_14"] / g["Close"]

    # =========================
    # (C) Estresse / cauda: Max Drawdown 30d
    # =========================
    g["max_dd_30d"] = rolling_max_drawdown(g["Close"], window=30)

    # Retorna só o necessário (padroniza merge depois)
    out = g[["Date", "Ticker", "ret_1d", "vol_7d", "vol_30d", "atrp_14", "max_dd_30d"]].copy()
    return out


# =========================
# 3) Leitura e preparação
# =========================
df = pd.read_csv(INPUT_CSV)

# Ajuste de tipos e ordenação
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])

# Checagens mínimas (ajuda a debugar se o CSV vier diferente)
required_cols = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Faltando colunas no CSV: {missing}. Colunas atuais: {list(df.columns)}")


# =========================
# 4) Cálculo por ticker
# =========================
df_vol = df.groupby("Ticker", group_keys=False).apply(compute_vol_features)

# =========================
# 5) Limpeza e export
# =========================
# Remove linhas iniciais sem dados suficientes (janelas rolling)
df_vol = df_vol.dropna().reset_index(drop=True)

# Garante que a pasta features existe
os.makedirs("features", exist_ok=True)

df_vol.to_csv(OUTPUT_CSV, index=False)

print("OK! features/features_vol.csv gerado.")
print(df_vol.head())
print("Tickers:", df_vol["Ticker"].unique())