import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, 
    classification_report, confusion_matrix
)

def calcular_benchmarks_fuzzy_direto():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Apontando para a sua pasta Fuzzy onde o 'close' já existe
    pasta_base = os.path.join(script_dir, 'dados_normalizados', 'fuzzy', 'train')
    
    if not os.path.exists(pasta_base):
        print(f"❌ Pasta não encontrada: {pasta_base}")
        return

    # Busca especificamente os arquivos fuzzy
    arquivos = [f for f in os.listdir(pasta_base) if f.endswith('_fuzzy_train.csv')]
    horizontes = [1, 7, 30]

    arquivo_out = "benchmarks_fuzzy_detalhados.txt"
    if os.path.exists(arquivo_out): os.remove(arquivo_out)

    print(f"🚀 Rodando Baselines diretamente nos arquivos Fuzzy ({len(arquivos)} ativos)...")

    for arquivo in arquivos:
        ticker = arquivo.split('_')[0]
        df = pd.read_csv(os.path.join(pasta_base, arquivo))
        
        # Verificação de segurança para a coluna close
        if 'close' not in df.columns:
            print(f"⚠️ Pulo: Coluna 'close' não encontrada em {arquivo}")
            continue

        for h in horizontes:
            # --- 1. Preparação do Target (Futuro) ---
            df_temp = df.copy()
            df_temp['ret_futuro'] = df_temp['close'].pct_change(h).shift(-h)
            df_temp = df_temp.dropna(subset=['ret_futuro'])
            
            if len(df_temp) < 100: continue
            limite = int(len(df_temp) * 0.8)
            
            # Discretização baseada nos quantis do treino (80%)
            _, bins = pd.qcut(df_temp['ret_futuro'].iloc[:limite], 3, retbins=True, labels=[0, 1, 2])
            bins[0], bins[-1] = -np.inf, np.inf
            
            y_train = pd.cut(df_temp['ret_futuro'].iloc[:limite], bins=bins, labels=[0, 1, 2]).astype(int)
            y_test = pd.cut(df_temp['ret_futuro'].iloc[limite:], bins=bins, labels=[0, 1, 2]).astype(int)

            # --- 2. BASELINE MAJORITÁRIO ---
            classe_maj = y_train.mode()[0]
            preds_maj = np.full(len(y_test), classe_maj)
            
            probs_maj = np.zeros((len(y_test), 3))
            probs_maj[:, classe_maj] = 1.0
            loss_maj = log_loss(y_test, probs_maj, labels=[0, 1, 2])

            # --- 3. BASELINE PERSISTÊNCIA (Random Walk) ---
            # Assume que o sinal de t+h será igual ao sinal de t
            df_temp['ret_passado'] = df_temp['close'].pct_change(h)
            df_temp['sinal_hoje'] = pd.cut(df_temp['ret_passado'], bins=bins, labels=[0, 1, 2])
            preds_pers = df_temp['sinal_hoje'].iloc[limite:].fillna(classe_maj).astype(int)
            
            probs_pers = np.zeros((len(y_test), 3))
            for i, p in enumerate(preds_pers):
                probs_pers[i, p] = 1.0
            loss_pers = log_loss(y_test, probs_pers, labels=[0, 1, 2])

            # --- 4. SALVAR NO TXT ---
            with open(arquivo_out, "a") as f:
                for nome, p, loss in [("Majoritário", preds_maj, loss_maj), ("Persistência", preds_pers, loss_pers)]:
                    f.write(f"\n{'='*50}\nBASELINE FUZZY {nome.upper()}: {ticker} H{h}\n{'='*50}\n")
                    f.write(f"Acurácia: {accuracy_score(y_test, p):.2%}\n")
                    f.write(f"F1-Score Macro: {f1_score(y_test, p, average='macro'):.4f}\n")
                    f.write(f"Log Loss: {loss:.4f}\n")
                    f.write(f"\nRelatório de Classificação:\n{classification_report(y_test, p, zero_division=0)}\n")
                    f.write(f"\nMatriz de Confusão:\n{confusion_matrix(y_test, p)}\n\n")

    print(f"🏁 Concluído! Resultados em: {arquivo_out}")

if __name__ == "__main__":
    calcular_benchmarks_fuzzy_direto()