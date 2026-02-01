import os
import pandas as pd


# Config
TREND_CSV = "features/features_trend.csv"
VOL_CSV = "features/features_vol.csv"
RELQUAL_CSV = "features/features_relqual.csv"

OUTPUT_CSV = "features/features_normal_all.csv"


def load_and_check(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    # 1) Checa colunas-chave
    required = {"Date", "Ticker"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{name} não tem as colunas obrigatórias {required}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    # 2) Padroniza tipos
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str)

    # 3) Ordena
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 4) Garante unicidade por (Date, Ticker)
    dup = df.duplicated(["Date", "Ticker"]).sum()
    if dup > 0:
        # se tiver duplicado, isso é erro de pipeline (ou gerou linhas repetidas)
        # aqui vou derrubar com mensagem para você corrigir na origem
        raise ValueError(f"{name} tem {dup} linhas duplicadas para (Date, Ticker).")

    return df


def main():
    df_trend = load_and_check(TREND_CSV, "TREND")
    df_vol = load_and_check(VOL_CSV, "VOL")
    df_rel = load_and_check(RELQUAL_CSV, "RELQUAL")

    # (Opcional) checar interseção de datas/linhas
    print("Linhas:")
    print("trend:", len(df_trend))
    print("vol  :", len(df_vol))
    print("rel  :", len(df_rel))

    # Merge 
    df_all = (
        df_trend.merge(df_vol, on=["Date", "Ticker"], how="inner")
                .merge(df_rel, on=["Date", "Ticker"], how="inner")
    )

    # Ordena e salva
    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    os.makedirs("features", exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"\nOK! Dataset final salvo em: {OUTPUT_CSV}")
    print("Shape final:", df_all.shape)
    print(df_all.head())


if __name__ == "__main__":
    main() 
