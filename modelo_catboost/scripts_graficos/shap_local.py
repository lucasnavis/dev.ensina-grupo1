import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from matplotlib.colors import LinearSegmentedColormap

# --- CONFIGURAÇÕES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "models")
PASTA_DADOS_RAIZ = os.path.join(SCRIPT_DIR, "dados_normalizados")
PASTA_SAIDA = os.path.join(SCRIPT_DIR, "shap_analise_local")
ULTIMAS_N = 10  
os.makedirs(PASTA_SAIDA, exist_ok=True)

cores_ambar_cinza = ["#4D4D4D", "#D3D3D3", "#FFCC00"]
cmap_final = LinearSegmentedColormap.from_list("AmbarCinza", cores_ambar_cinza)

def carregar_dados(tipo, ativo):
    sub = "normal" if tipo == "NORMAL" else "fuzzy"
    path = os.path.join(PASTA_DADOS_RAIZ, sub, "test")
    if not os.path.exists(path): return None
    for f in os.listdir(path):
        if ativo.upper() in f.upper() and f.endswith(".csv"):
            return pd.read_csv(os.path.join(path, f)).select_dtypes(include=[np.number])
    return None

# --- PROCESSAMENTO ---
arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.endswith('.cbm')]
pares = {}
for f in arquivos:
    p = f.replace(".cbm", "").split("_")
    if len(p) >= 4:
        chave = (p[2], p[3]) 
        if chave not in pares: pares[chave] = {}
        pares[chave][p[1]] = f

for (ativo, horiz), mods in pares.items():
    pasta_ativo = os.path.join(PASTA_SAIDA, f"{ativo}_{horiz}")
    os.makedirs(pasta_ativo, exist_ok=True)

    for tipo in ["NORMAL", "FUZZY"]:
        try:
            if tipo not in mods: continue
            df_raw = carregar_dados(tipo, ativo)
            if df_raw is None: continue
            
            model = CatBoostClassifier().load_model(os.path.join(PASTA_MODELOS, mods[tipo]))
            cols = model.feature_names_
            
            # 1. Pegamos as últimas N velas e INVERTEMOS a ordem das linhas
            # Agora a primeira linha é a vela ATUAL (T-0)
            X_local = df_raw[cols].tail(ULTIMAS_N).iloc[::-1]
            
            # 2. Calculamos o SHAP (os valores SHAP virão na ordem T-0, T-1... T-9)
            sv = model.get_feature_importance(Pool(X_local), type='ShapValues', thread_count=-1)
            
            # 3. Extração correta dos valores (ajuste de dimensão)
            if len(sv.shape) == 3:
                if sv.shape[1] == len(cols) + 1: # [Amostra, Feat+Bias, Classe]
                    vals = sv[:, :len(cols), 1]
                else: # [Amostra, Classe, Feat+Bias]
                    vals = sv[:, 1, :len(cols)]
            else:
                vals = sv[:, :len(cols)]

            # 4. Criamos labels fixos de T-0 (atual) até T-9 (passado)
            labels_temporais = [f"T-{i}" for i in range(ULTIMAS_N)]
            
            # 5. Criamos o DataFrame. Agora a ordem cronológica está correta:
            # A primeira coluna do gráfico será o Agora (T-0)
            df_plot = pd.DataFrame(vals, columns=cols, index=labels_temporais)
            
            if tipo == "FUZZY":
                df_plot = df_plot.groupby(lambda x: x.split('_')[0], axis=1).sum()

            # --- PLOTAGEM ---
            plt.figure(figsize=(14, 8), facecolor='white')
            sns.heatmap(df_plot.T, 
                        annot=True, 
                        cmap=cmap_final, 
                        center=0, 
                        fmt='.4f', 
                        cbar_kws={'label': 'Peso SHAP'},
                        annot_kws={"size": 9, "weight": "bold"})
            
            plt.title(f"Impacto Temporal SHAP: {ativo} {horiz} ({tipo})\n(Esquerda: ATUAL T-0 | Direita: PASSADO T-9)", 
                      fontsize=14, fontweight='bold', pad=20)
            plt.xlabel("Linha do Tempo (Decrescente do Presente para o Passado)", fontsize=11, fontweight='bold')
            plt.ylabel("Variáveis", fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_ativo, f"shap_local_{tipo}.png"), dpi=120)
            plt.close('all')
            print(f" [OK] {ativo}_{horiz} ({tipo})")

        except Exception as e:
            print(f" [ERRO] {ativo} {tipo}: {e}")

print(f"\n Concluído!")