import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURAÇÕES DE CAMINHOS ---
datasets = {
    "normal": {
        "input": r'C:\Users\anapa\OneDrive\Área de Trabalho\projeto_alocacao_fuzzy\scripts\features_normal_dataset_with_close.csv',
        "output": r'C:\Users\anapa\OneDrive\Área de Trabalho\projeto_alocacao_fuzzy\scripts\dados_normalizados\normal',
        "cols": ['close', 'mom_7d', 'vov_60d', 'dd_dur_60d', 'autocorr_30d', 'fgi_vol_30d']
    },
    "fuzzy": {
        "input": r'C:\Users\anapa\OneDrive\Área de Trabalho\projeto_alocacao_fuzzy\scripts\features_fuzzy_dataset_with_close.csv',
        "output": r'C:\Users\anapa\OneDrive\Área de Trabalho\projeto_alocacao_fuzzy\scripts\dados_normalizados\fuzzy',
        "cols": ['close', 'trend_bear', 'trend_neutral', 'trend_bull', 
    'vol_low', 'vol_mid', 'vol_high', 
    'stress_low', 'stress_mid', 'stress_high', 
    'ac_negative', 'ac_neutral', 'ac_positive', 
    'macro_calm', 'macro_neutral', 'macro_stress'
        ]
    }
}

def processar_datasets():
    for tipo, config in datasets.items():
        print(f"\nIniciando processamento: {tipo.upper()}")
        
        if not os.path.exists(config["output"]):
            import os
            from pathlib import Path
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split


            # Local do script
            script_dir = Path(__file__).resolve().parent
            # Repo root (assume pasta modelo_catboost está no root)
            repo_root = script_dir.parent

            # Entradas esperadas no repositório (relativas ao root)
            in_normal = repo_root / '..' / 'features + correl' / 'final' / 'features_normal_dataset_with_close.csv'
            in_fuzzy = repo_root / '..' / 'features + correl' / 'final' / 'features_fuzzy_dataset_with_close.csv'

            out_base = script_dir / 'dados_normalizados'

            datasets = {
                'normal': {
                    'input': str(in_normal.resolve()),
                    'output': str((out_base / 'normal').resolve()),
                    'cols': ['close', 'mom_7d', 'vov_60d', 'dd_dur_60d', 'autocorr_30d', 'fgi_vol_30d']
                },
                'fuzzy': {
                    'input': str(in_fuzzy.resolve()),
                    'output': str((out_base / 'fuzzy').resolve()),
                    'cols': [
                        'close',
                        'trend_bear', 'trend_neutral', 'trend_bull',
                        'vol_low', 'vol_mid', 'vol_high',
                        'stress_low', 'stress_mid', 'stress_high',
                        'ac_negative', 'ac_neutral', 'ac_positive',
                        'macro_calm', 'macro_neutral', 'macro_stress'
                    ]
                }
            }


            def processar_datasets():
                for tipo, config in datasets.items():
                    print(f"\nIniciando processamento: {tipo.upper()}")

                    input_path = Path(config['input'])
                    output_base = Path(config['output'])

                    if not input_path.exists():
                        print(f"  ERRO: arquivo de entrada não encontrado: {input_path}")
                        continue

                    output_base.mkdir(parents=True, exist_ok=True)

                    df = pd.read_csv(input_path)

                    # Normaliza nomes de colunas para lowercase (robustez)
                    df.columns = df.columns.str.strip().str.lower()

                    if 'ticker' not in df.columns:
                        # tenta detectar coluna ticker em variações
                        candidates = [c for c in df.columns if 'tick' in c]
                        if candidates:
                            df = df.rename(columns={candidates[0]: 'ticker'})

                    tickers = df['ticker'].unique()

                    for t in tickers:
                        try:
                            df_temp = df[df['ticker'] == t].copy()

                            cols = [c for c in config['cols'] if c in df_temp.columns]
                            if not cols:
                                print(f"  Aviso: nenhuma coluna prevista encontrada para {t} (tipo={tipo}). Pulando.")
                                continue

                            df_temp[cols] = df_temp[cols].apply(pd.to_numeric, errors='coerce')
                            df_temp[cols] = df_temp[cols].ffill().fillna(0)

                            train_df, test_df = train_test_split(df_temp, test_size=0.2, shuffle=False)

                            scaler = MinMaxScaler()
                            train_scaled = train_df.copy()
                            train_scaled[cols] = scaler.fit_transform(train_df[cols])

                            test_scaled = test_df.copy()
                            test_scaled[cols] = scaler.transform(test_df[cols])

                            for modo, data in [('train', train_scaled), ('test', test_scaled)]:
                                pasta_modo = output_base / modo
                                pasta_modo.mkdir(parents=True, exist_ok=True)
                                caminho_final = pasta_modo / f'{t}_{tipo}_{modo}.csv'
                                data.to_csv(caminho_final, index=False)

                            print(f"  ✔ {t}: Processado (Treino: {len(train_scaled)} | Teste: {len(test_scaled)})")

                        except Exception as e:
                            print(f"  Erro em {t}: {e}")


            if __name__ == '__main__':
                processar_datasets()
                print('\n Todos os datasets foram normalizados e divididos!')