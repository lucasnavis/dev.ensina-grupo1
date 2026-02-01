import os
import pandas as pd
from urllib.request import urlopen
import json


# CONFIG
FGI_API_URL = "https://api.alternative.me/fng/?limit=0"
OHLCV_CSV = "data/crypto_5y_ohlcv.csv"

FGI_CSV = "data/fgi.csv"                     
OUT_CSV = "features/features_fgi_drawdown.csv"



# FGI: baixar + carregar
def baixar_fgi_para_data_csv(dest_path=FGI_CSV):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with urlopen(FGI_API_URL) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    rows = payload.get("data", [])
    if not rows:
        raise ValueError("API do FGI retornou vazio.")

    df = pd.DataFrame(rows)

    # API vem com timestamp (segundos) + value (0-100)
    df["Date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.date.astype(str)
    df["fgi"] = df["value"].astype(float)

    out = df[["Date", "fgi"]].drop_duplicates("Date").sort_values("Date")
    out.to_csv(dest_path, index=False)

    print(f"OK! Baixei e salvei FGI em: {dest_path} | linhas: {len(out)}")


def load_fgi() -> pd.DataFrame:
    # se não existir, baixa automaticamente
    if not os.path.exists(FGI_CSV):
        baixar_fgi_para_data_csv(FGI_CSV)

    fgi = pd.read_csv(FGI_CSV)

    # garante colunas padrão: Date e fgi
    if "Date" not in fgi.columns or "fgi" not in fgi.columns:
        # tenta detectar automaticamente (caso alguém mude o arquivo)
        date_col = None
        for c in ["Date", "date", "DATA", "data", "timestamp", "Time", "time"]:
            if c in fgi.columns:
                date_col = c
                break
        if date_col is None:
            raise ValueError(f"FGI CSV sem coluna de data. Colunas: {list(fgi.columns)}")

        val_col = None
        for c in ["fgi", "FGI", "value", "Value", "fear_greed", "fearGreed", "index", "Index"]:
            if c in fgi.columns:
                val_col = c
                break
        if val_col is None:
            raise ValueError(f"FGI CSV sem coluna do valor. Colunas: {list(fgi.columns)}")

        fgi = fgi[[date_col, val_col]].copy()
        fgi.rename(columns={date_col: "Date", val_col: "fgi"}, inplace=True)

    fgi["Date"] = pd.to_datetime(fgi["Date"])
    fgi = fgi.sort_values("Date").drop_duplicates("Date")
    return fgi



# Feature: drawdown 90d
def feature_drawdown_90d(close: pd.Series, janela: int = 90) -> pd.Series:
    max_rol = close.rolling(janela, min_periods=janela).max()
    return (close - max_rol) / max_rol  # <= 0 (ex: -0.25 = -25%)



# MAIN
def main():
    df = pd.read_csv(OHLCV_CSV)

    required = {"Date", "Ticker", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV CSV sem colunas: {missing}. Colunas atuais: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df["drawdown_90d"] = df.groupby("Ticker")["Close"].transform(lambda s: feature_drawdown_90d(s, 90))

    fgi = load_fgi()
    df = df.merge(fgi, on="Date", how="left")

    out = df[["Date", "Ticker", "drawdown_90d", "fgi"]].dropna().reset_index(drop=True)

    os.makedirs("features", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("OK! Gerou:", OUT_CSV)
    print("Shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    main()
