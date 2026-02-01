import pandas as pd
from pathlib import Path

BASE_OHLC = Path("data/crypto_5y_ohlcv.csv")
NORMAL_IN = Path("final/features_normal_datasetfinal.csv")
FUZZY_IN  = Path("final/features_fuzzy_datasetfinal.csv")

NORMAL_OUT = Path("final/features_normal_dataset_with_close.csv")
FUZZY_OUT  = Path("final/features_fuzzy_dataset_with_close.csv")


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _standardize_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)

    # date
    if "date" not in df.columns:
        raise ValueError(f"Não achei coluna de data no OHLC ({path}). Colunas: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # ticker/symbol
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()
    elif "symbol" in df.columns:
        df["ticker"] = df["symbol"].astype(str).str.split("/").str[0].str.strip()
    else:
        raise ValueError(f"Não achei coluna Ticker/Symbol no OHLC ({path}). Colunas: {list(df.columns)}")

    # close
    if "close" not in df.columns:
        raise ValueError(f"Não achei coluna close no OHLC ({path}). Colunas: {list(df.columns)}")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # mantém só o necessário (e remove duplicatas por segurança)
    df = df[["date", "ticker", "close"]].drop_duplicates(subset=["date", "ticker"])
    return df


def _add_close(features_path: Path, out_path: Path, ohlc: pd.DataFrame) -> None:
    df = pd.read_csv(features_path)
    df = _normalize_cols(df)

    # garante que o seu dataset tem date/ticker
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(f"{features_path} precisa ter colunas Date e Ticker. Colunas: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.strip()

    merged = df.merge(ohlc, on=["date", "ticker"], how="left")

    # coloca close logo depois de ticker
    cols = list(merged.columns)
    if "close" in cols:
        cols.remove("close")
        ticker_i = cols.index("ticker")
        cols.insert(ticker_i + 1, "close")
        merged = merged[cols]

    merged.to_csv(out_path, index=False)

    # debug rápido
    pct_missing = merged["close"].isna().mean() * 100
    print(f"OK! Salvo: {out_path}")
    print(f"Linhas: {len(merged)} | % close faltando: {pct_missing:.2f}%")
    if pct_missing > 0:
        print("Se ficou alto, é quase sempre mismatch de Ticker (ex: ADA vs ADA/USDT) ou Date.")


def main():
    ohlc = _standardize_ohlc(BASE_OHLC)

    _add_close(NORMAL_IN, NORMAL_OUT, ohlc)
    _add_close(FUZZY_IN, FUZZY_OUT, ohlc)


if __name__ == "__main__":
    main()
