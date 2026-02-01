import os
import numpy as np
import pandas as pd

# ======================
# CONFIG
# ======================
INPUT_CSV = "features/features_base_all.csv"
OUTPUT_CSV = "final/features_normal_datasetfinal.csv"

LIMITE_MAX_CORR = 0.25  # você pode trocar para 0.05 se quiser

# colunas que nunca entram como feature
DROP_ALWAYS = ["Date", "Ticker", "ret_1d"]

# ======================
# LOAD
# ======================
df = pd.read_csv(INPUT_CSV)

# separa colunas "id"
base_cols = [c for c in ["Date", "Ticker"] if c in df.columns]

# dataframe só com features numéricas
df_feat = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], errors="ignore")
df_feat = df_feat.select_dtypes(include="number")

if df_feat.shape[1] < 2:
    raise ValueError("Poucas features numéricas para calcular correlação (precisa >= 2).")

# ======================
# CORRELATION (ABS)
# ======================
corr_abs = df_feat.corr().abs()

# ✅ FIX: cria matriz gravável e zera diagonal
corr_mat = corr_abs.to_numpy(copy=True)
np.fill_diagonal(corr_mat, 0.0)

# máximo |corr| de cada feature com as outras
max_corr_each = corr_mat.max(axis=0)

# features que passam no filtro
keep_features = df_feat.columns[max_corr_each <= LIMITE_MAX_CORR].tolist()

# ======================
# OUTPUT
# ======================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df_out = df[base_cols + keep_features].copy()
df_out.to_csv(OUTPUT_CSV, index=False)

print("OK! Dataset filtrado salvo em:", OUTPUT_CSV)
print("Limite (max |corr|):", LIMITE_MAX_CORR)
print("Features selecionadas:", len(keep_features))
print("Shape final:", df_out.shape)
print("Lista de features:", keep_features)
