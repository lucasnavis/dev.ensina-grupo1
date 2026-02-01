import os
import numpy as np
import pandas as pd

# =====================
# CONFIG
# =====================
INPUT_CSV = "features/features_base_all.csv"
OUTPUT_CSV = "final/features_normal_datasetfinal.csv"

LIMITE_CORR = 0.25
BASE_COLS = ["Date", "Ticker"]
DROP_ALWAYS = ["ret_1d"]

# blocos (precisamos manter >= 1 por bloco)
GROUPS = {
    # trend (tendência)
    "trend": [
        "mom_7d", "mom_14d", "mom_30d", "mom_60d",
        "ema_diff", "ema_cross_12_26",
        "slope_30d",
        "rsi_14_n",
    ],
    # vol (volatilidade)
    "vol": [
        "vol_7d", "vol_30d", "vol_60d",
        "idio_vol_30d",
        "atrp_14",
        "rv_14d", "rv_60d",
        "downside_vol_60d",
        "parkinson_vol_30d",
        "vov_60d",
    ],
    # stress (stress/tail risk)
    "stress": [
        "drawdown_90d", "max_dd_30d",
        "max_loss_30d",
        "cvar_5_60d",
        "dd_dur_60d",
    ],
    # quality (qualidade relativa / “qualidade do retorno”)
    "quality": [
        "ir_30d", "rs_30d",
        "sharpe_60d", "sortino_60d",
        "hit_ratio_30d",
        "profit_factor_60d",
        "omega_60d",
        "autocorr_30d",
        "skew_60d", "kurt_60d",
    ],
    # macro (FGI e derivados)
    "macro": [
        "fgi",
        "fgi_chg_1d", "fgi_chg_7d", "fgi_chg_14d",
        "fgi_ema_14", "fgi_gap_ema14",
        "fgi_z_90d",
        "fgi_vol_30d",
        "fgi_rank_180d",
        # se ainda existir no seu dataset antigo:
        "fgi_y",
    ],
}

# plano B (só se macro falhar)
FORCE_KEEP_IF_NEEDED = ["fgi"]


# =====================
def corr_abs_matrix(df_features: pd.DataFrame) -> pd.DataFrame:
    corr = df_features.corr(method="pearson")
    corr_abs = corr.abs().copy()

    arr = corr_abs.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=corr_abs.index, columns=corr_abs.columns)


def max_corr_with_selected(corr_abs: pd.DataFrame, feat: str, selected: list[str]) -> float:
    if not selected:
        return 0.0
    if feat not in corr_abs.index:
        return 999.0
    cols = [c for c in selected if c in corr_abs.columns]
    if not cols:
        return 0.0
    return float(corr_abs.loc[feat, cols].max())


def pick_one_from_group(corr_abs, candidates, selected, limit):
    if not candidates:
        return None, []

    candidates_sorted = sorted(
        candidates,
        key=lambda c: max_corr_with_selected(corr_abs, c, selected),
    )

    for c in candidates_sorted:
        if max_corr_with_selected(corr_abs, c, selected) <= limit:
            return c, candidates_sorted

    return None, candidates_sorted


def main():
    df = pd.read_csv(INPUT_CSV)

    base_cols = [c for c in BASE_COLS if c in df.columns]

    df_features = (
        df.drop(columns=base_cols, errors="ignore")
        .select_dtypes(include="number")
        .copy()
    )

    # remove colunas sempre descartadas
    for c in DROP_ALWAYS:
        if c in df_features.columns:
            df_features = df_features.drop(columns=[c])

    # remove colunas 100% NaN (correlação quebra/atrapalha)
    df_features = df_features.dropna(axis=1, how="all")

    corr_abs = corr_abs_matrix(df_features)

    selected = []
    chosen_by_group = {}
    failed_groups = {}

    # tenta 1 por grupo
    for gname, candidates in GROUPS.items():
        candidates = [c for c in candidates if c in df_features.columns and c not in selected]
        chosen, ordered = pick_one_from_group(corr_abs, candidates, selected, LIMITE_CORR)

        if chosen is None:
            failed_groups[gname] = ordered
        else:
            selected.append(chosen)
            chosen_by_group[gname] = chosen

    # plano B: só se macro falhar
    if "macro" not in chosen_by_group and FORCE_KEEP_IF_NEEDED:
        forced = [c for c in FORCE_KEEP_IF_NEEDED if c in df_features.columns and c not in selected]
        if forced:
            selected.append(forced[0])
            chosen_by_group["macro_forced"] = forced[0]

    df_out = pd.concat([df[base_cols], df_features[selected]], axis=1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("OK! Dataset filtrado salvo.")
    print("Input:", INPUT_CSV)
    print("Output:", OUTPUT_CSV)
    print("Regra: max |corr| <=", LIMITE_CORR)
    print("Escolhidas por grupo:", chosen_by_group)
    print("Mantidas:", selected)
    print("Shape final:", df_out.shape)

    if failed_groups:
        print("\nGrupos que falharam (nenhum candidato passou no limite):")
        for g, ordered in failed_groups.items():
            print(f"- {g}: {ordered}")


if __name__ == "__main__":
    main()
