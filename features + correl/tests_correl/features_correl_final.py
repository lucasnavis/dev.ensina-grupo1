import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_CSV = "final/features_normal_datasetfinal.csv"
OUT_DIR = "tests_correl"
LIMIT = 0.1  # só para visualização (cinza quando |corr| <= LIMIT)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)

# Date e Ticker não entram na correlação
df_features = df.drop(columns=["Date", "Ticker"], errors="ignore")

# Mantém apenas colunas numéricas
df_features = df_features.select_dtypes(include="number")

# Pearson
corr = df_features.corr()

corr.to_csv(os.path.join(OUT_DIR, "correlation_matrix_final.csv"))

# Máscara para "zona neutra" (|corr| <= LIMIT) ficar cinza
neutral_mask = corr.abs() <= LIMIT

# Heatmap base (cores)
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

# Overlay cinza nas células neutras
sns.heatmap(
    corr,
    mask=~neutral_mask,     # só desenha onde é neutro
    annot=False,
    cmap=sns.color_palette(["#B0B0B0"], as_cmap=True),
    cbar=False,
    linewidths=0.5
)

plt.title(f"Correlation Heatmap - FINAL (|corr| <= {LIMIT:.2f} em cinza)")
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap_final.png"), dpi=200)
plt.show()

print("OK! Salvei:")
print(f"- {OUT_DIR}/correlation_matrix_final.csv")
print(f"- {OUT_DIR}/correlation_heatmap_final.png")
