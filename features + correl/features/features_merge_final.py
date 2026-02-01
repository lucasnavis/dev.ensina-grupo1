import os
import pandas as pd

NORMAL_CSV = "features/features_normal_all.csv"
FGI_DD_CSV = "features/features_fgi_drawdown.csv"
EXTRA_CSV  = "features/features_extra.csv"

OUT_CSV = "features/features_base_all.csv"
KEY = ["Date", "Ticker"]

def main():
    # 1) Load
    df_normal = pd.read_csv(NORMAL_CSV)
    df_fgi    = pd.read_csv(FGI_DD_CSV)
    df_extra  = pd.read_csv(EXTRA_CSV)

    # 2) Garantir tipo Date igual nos três
    for df in (df_normal, df_fgi, df_extra):
        df["Date"] = pd.to_datetime(df["Date"])

    # 3) Merge (inner = só datas/tickers que existem nos 3)
    df_all = df_normal.merge(df_fgi, on=KEY, how="inner")
    df_all = df_all.merge(df_extra, on=KEY, how="inner")

    # (Opcional) Se tiver colunas duplicadas além de Date/Ticker, resolve aqui:
    # df_all = df_all.loc[:, ~df_all.columns.duplicated()]

    # 4) Ordena e salva
    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    os.makedirs("features", exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)

    print(f"OK! Dataset final salvo em: {OUT_CSV}")
    print("Shape final:", df_all.shape)
    print(df_all.head())
    print("Colunas:", list(df_all.columns))

if __name__ == "__main__":
    main()
