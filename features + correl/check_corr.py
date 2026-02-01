import pandas as pd
import numpy as np

df = pd.read_csv('features/features_base_all.csv')
df_feat = df.drop(columns=['Date', 'Ticker', 'ret_1d'], errors='ignore').select_dtypes(include='number')

corr_abs = df_feat.corr().abs()
corr_mat = corr_abs.to_numpy(copy=True)
np.fill_diagonal(corr_mat, 0.0)

max_corr_each = corr_mat.max(axis=0)

print('Max correlations per feature (lowest 15):')
for col, val in sorted(zip(df_feat.columns, max_corr_each), key=lambda x: x[1])[:15]:
    print(f'{col}: {val:.4f}')

print('\nMin max correlation:', max_corr_each.min())
print('Median max correlation:', np.median(max_corr_each))
