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
            os.makedirs(config["output"])

        df = pd.read_csv(config["input"])
        tickers = df['ticker'].unique()

        for t in tickers:
            try:
                # 1. Filtragem e Limpeza
                df_temp = df[df['ticker'] == t].copy()
                df_temp[config["cols"]] = df_temp[config["cols"]].apply(pd.to_numeric, errors='coerce')
                
                # Preencher NaNs para evitar erro no Scaler
                df_temp[config["cols"]] = df_temp[config["cols"]].ffill().fillna(0)

                # 2. Divisão Treino/Teste (80% treino, 20% teste)
                # IMPORTANTE: shuffle=False para Séries Temporais
                train_df, test_df = train_test_split(df_temp, test_size=0.2, shuffle=False)

                # 3. Normalização Profissional
                scaler = MinMaxScaler()

                # Ajusta APENAS no treino
                train_scaled = train_df.copy()
                train_scaled[config["cols"]] = scaler.fit_transform(train_df[config["cols"]])

                # Aplica no teste usando a régua do treino (Evita Leakage)
                test_scaled = test_df.copy()
                test_scaled[config["cols"]] = scaler.transform(test_df[config["cols"]])

                # 4. Salvamento Organizado
                # Criar subpastas para treino e teste
                for modo, data in [("train", train_scaled), ("test", test_scaled)]:
                    pasta_modo = os.path.join(config["output"], modo)
                    if not os.path.exists(pasta_modo): os.makedirs(pasta_modo)
                    
                    caminho_final = os.path.join(pasta_modo, f'{t}_{tipo}_{modo}.csv')
                    data.to_csv(caminho_final, index=False)

                print(f"  ✔ {t}: Processado (Treino: {len(train_scaled)} | Teste: {len(test_scaled)})")

            except Exception as e:
                print(f"Erro em {t}: {e}")

processar_datasets()
print("\n Todos os datasets foram normalizados e divididos sem vazamento de dados!")