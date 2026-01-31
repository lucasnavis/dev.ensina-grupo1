import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, 
    classification_report, confusion_matrix
)

def calcular_benchmarks_normal():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Pasta de dados puramente normais
    pasta_base = os.path.join(script_dir, 'dados_normalizados', 'normal', 'train')
    
    if not os.path.exists(pasta_base):
        print(f"❌ Pasta não encontrada: {pasta_base}")
        return

    arquivos = [f for f in os.listdir(pasta_base) if f.endswith('_normal_train.csv')]
    horizontes = [1, 7, 30]

    # Arquivo de saída específico para o baseline normal
    arquivo_out = "benchmarks_normal_detalhados.txt"
    if os.path.exists(arquivo_out): os.remove(arquivo_out)

    print(f"🚀 Calculando Baselines para {len(arquivos)} ativos (Pipeline Normal)...")

    for arquivo in arquivos:
        ticker = arquivo.split('_')[0]
        df = pd.read_csv(os.path.join(pasta_base, arquivo))
        
        for h in horizontes:
            # --- 1. Preparação do Target ---
            df_temp = df.copy()
            df_temp['ret_futuro'] = df_temp['close'].pct_change(h).shift(-h)
            df_temp = df_temp.dropna(subset=['ret_futuro'])
            
            if len(df_temp) < 100: continue
            limite = int(len(df_temp) * 0.8)
            
            # Discretização baseada nos quantis do treino (Igual ao Fuzzy)
            _, bins = pd.qcut(df_temp['ret_futuro'].iloc[:limite], 3, retbins=True, labels=[0, 1, 2])
            bins[0], bins[-1] = -np.inf, np.inf
            
            y_train = pd.cut(df_temp['ret_futuro'].iloc[:limite], bins=bins, labels=[0, 1, 2]).astype(int)
            y_test = pd.cut(df_temp['ret_futuro'].iloc[limite:], bins=bins, labels=[0, 1, 2]).astype(int)

            # --- 2. BASELINE MAJORITÁRIO ---
            classe_maj = y_train.mode()[0]
            preds_maj = np.full(len(y_test), classe_maj)
            
            # Probabilidades determinísticas (1.0 para a classe majoritária)
            probs_maj = np.zeros((len(y_test), 3))
            probs_maj[:, classe_maj] = 1.0
            
            loss_maj = log_loss(y_test, probs_maj, labels=[0, 1, 2])

            # --- 3. BASELINE PERSISTÊNCIA (Random Walk) ---
            df_temp['ret_passado'] = df_temp['close'].pct_change(h)
            df_temp['sinal_hoje'] = pd.cut(df_temp['ret_passado'], bins=bins, labels=[0, 1, 2])
            preds_pers = df_temp['sinal_hoje'].iloc[limite:].fillna(classe_maj).astype(int)
            
            probs_pers = np.zeros((len(y_test), 3))
            for i, p in enumerate(preds_pers):
                probs_pers[i, p] = 1.0
            
            loss_pers = log_loss(y_test, probs_pers, labels=[0, 1, 2])

            # --- 4. SALVAR RESULTADOS ---
            with open(arquivo_out, "a") as f:
                for nome, p, loss in [("Majoritário", preds_maj, loss_maj), ("Persistência", preds_pers, loss_pers)]:
                    f.write(f"\n{'='*45}\nBASELINE NORMAL {nome.upper()}: {ticker} H{h}\n{'='*45}\n")
                    f.write(f"Acurácia: {accuracy_score(y_test, p):.2%}\n")
                    f.write(f"F1-Score Macro: {f1_score(y_test, p, average='macro'):.4f}\n")
                    f.write(f"Log Loss: {loss:.4f}\n")
                    f.write(f"\nMatriz de Confusão:\n{confusion_matrix(y_test, p)}\n")
                    f.write(f"\nRelatório de Classificação:\n{classification_report(y_test, p, zero_division=0)}\n")

    print(f"✅ Benchmarks Normais salvos em: {arquivo_out}")

if __name__ == "__main__":
    calcular_benchmarks_normal()