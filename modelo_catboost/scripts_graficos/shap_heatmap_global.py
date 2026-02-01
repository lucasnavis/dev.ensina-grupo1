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
PASTA_SAIDA_GLOBAL = os.path.join(SCRIPT_DIR, "shap_analise_local_global")
ULTIMAS_N = 10  
os.makedirs(PASTA_SAIDA_GLOBAL, exist_ok=True)

# Paleta Amarela Premium para consistência com suas matrizes
cores_ambar_cinza = ["#4D4D4D", "#D3D3D3", "#FFD700"]
cmap_final = LinearSegmentedColormap.from_list("AmbarCinza", cores_ambar_cinza)

def carregar_dados(tipo, ativo):
    sub = "normal" if tipo == "NORMAL" else "fuzzy"
    path = os.path.join(PASTA_DADOS_RAIZ, sub, "test")
    if not os.path.exists(path): return None
    for f in os.listdir(path):
        if ativo.upper() in f.upper() and f.endswith(".csv"):
            return pd.read_csv(os.path.join(path, f)).select_dtypes(include=[np.number])
    return None

# --- ACUMULADORES GLOBAIS ---
acumulador_temporal = {"NORMAL": [], "FUZZY": []}

# --- PROCESSAMENTO ---
arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.endswith('.cbm')]
pares = {}
for f in arquivos:
    p = f.replace(".cbm", "").split("_")
    if len(p) >= 4:
        chave = (p[2], p[3]) 
        if chave not in pares: pares[chave] = {}
        pares[chave][p[1]] = f

print(f"Iniciando Consolidação Temporal Global para {len(pares)} pares...")

for (ativo, horiz), mods in pares.items():
    for tipo in ["NORMAL", "FUZZY"]:
        try:
            if tipo not in mods: continue
            df_raw = carregar_dados(tipo, ativo)
            if df_raw is None: continue
            
            model = CatBoostClassifier().load_model(os.path.join(PASTA_MODELOS, mods[tipo]))
            cols = model.feature_names_
            
            # Últimas N velas (T-0 é a mais recente)
            X_local = df_raw[cols].tail(ULTIMAS_N).iloc[::-1]
            sv = model.get_feature_importance(Pool(X_local), type='ShapValues', thread_count=-1)
            
            # Extração de valores (Classe 1 - Neutro/Tendência)
            if len(sv.shape) == 3:
                vals = sv[:, :len(cols), 1] if sv.shape[1] == len(cols) + 1 else sv[:, 1, :len(cols)]
            else:
                vals = sv[:, :len(cols)]

            labels_temporais = [f"T-{i}" for i in range(ULTIMAS_N)]
            df_temp = pd.DataFrame(vals, columns=cols, index=labels_temporais)
            
            if tipo == "FUZZY":
                df_temp = df_temp.groupby(lambda x: x.split('_')[0], axis=1).sum()

            acumulador_temporal[tipo].append(df_temp)

        except Exception as e:
            print(f" [ERRO] {ativo} {tipo}: {e}")

# --- GERAÇÃO DOS HEATMAPS GLOBAIS ---
for tipo in ["NORMAL", "FUZZY"]:
    if not acumulador_temporal[tipo]: continue
    
    # Média de impacto temporal de todos os ativos
    df_global = pd.concat(acumulador_temporal[tipo]).groupby(level=0).mean()
    
    # Selecionar as Top 15 variáveis com maior impacto absoluto médio para o gráfico não ficar gigante
    top_vars = df_global.abs().mean().sort_values(ascending=False).head(15).index
    df_plot = df_global[top_vars].T

    plt.figure(figsize=(16, 10), facecolor='white')
    sns.heatmap(df_plot, 
                annot=True, 
                cmap=cmap_final, 
                center=0, 
                fmt='.4f', 
                cbar_kws={'label': 'Impacto SHAP Médio Global'},
                annot_kws={"size": 10, "weight": "bold"})
    
    plt.title(f"MAPA TEMPORAL SHAP GLOBAL COM ESTABILIDADE TEMPORAL: MODELO {tipo}\n(Consolidado 10 Criptoativos | T-0 é o momento da previsão)", 
              fontsize=16, fontweight='bold', pad=25)
    plt.xlabel("Janelas Temporais (Passado <--- Presente)", fontsize=12, fontweight='bold')
    plt.ylabel("Variáveis/Indicadores", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA_GLOBAL, f"SHAP_TEMPORAL_GLOBAL_{tipo}.png"), dpi=300)
    plt.close()
    print(f" Heatmap Global {tipo} gerado.")

print(f"\n Pronto! Análise de memória temporal salva em: {PASTA_SAIDA_GLOBAL}")