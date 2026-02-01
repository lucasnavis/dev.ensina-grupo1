import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# --- CONFIGURAÇÕES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "models")
PASTA_DADOS_RAIZ = os.path.join(SCRIPT_DIR, "dados_normalizados")
PASTA_SAIDA_GLOBAL = os.path.join(SCRIPT_DIR, "shap_estabilidade_global")
N_JANELAS = 4 

CORES_ESTABILIDADE = ["#FFCC00", "#000000", "#4D4D4D", "#B8860B", "#A9A9A9"]

os.makedirs(PASTA_SAIDA_GLOBAL, exist_ok=True)

def carregar_dados(tipo, ativo):
    sub = "normal" if tipo == "NORMAL" else "fuzzy"
    path = os.path.join(PASTA_DADOS_RAIZ, sub, "test")
    if not os.path.exists(path): return None
    for f in os.listdir(path):
        if ativo.upper() in f.upper() and f.endswith(".csv"):
            return pd.read_csv(os.path.join(path, f)).select_dtypes(include=[np.number])
    return None

# --- ESTRUTURA PARA ACUMULAÇÃO GLOBAL ---
acumulador_global = {"NORMAL": [], "FUZZY": []}

arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.endswith('.cbm')]
pares = {}
for f in arquivos:
    p = f.replace(".cbm", "").split("_")
    if len(p) >= 4:
        chave = (p[2], p[3])
        if chave not in pares: pares[chave] = {}
        pares[chave][p[1]] = f

# --- PROCESSAMENTO ---
for (ativo, horiz), mods in pares.items():
    print(f"Processando Global: {ativo} | {horiz}")

    for tipo in ["NORMAL", "FUZZY"]:
        try:
            if tipo not in mods: continue
            df = carregar_dados(tipo, ativo)
            if df is None: continue
            
            model = CatBoostClassifier().load_model(os.path.join(PASTA_MODELOS, mods[tipo]))
            cols = model.feature_names_
            chunks = np.array_split(df[cols], N_JANELAS)
            
            janelas_ativo = []
            for i, chunk in enumerate(chunks):
                sv = model.get_feature_importance(Pool(chunk), type='ShapValues', thread_count=-1)
                val = np.abs(sv[:, :, :-1]).mean(axis=(0, 1)) if len(sv.shape)==3 else np.abs(sv[:, :-1]).mean(axis=0)
                
                s = pd.Series(val, index=cols)
                if tipo == "FUZZY": 
                    s = s.groupby(lambda x: x.split('_')[0]).sum()
                
                s.name = f"T{i+1}"
                janelas_ativo.append(s)

            # Transforma em DataFrame (Janelas x Features) e adiciona à lista global
            df_res_ativo = pd.concat(janelas_ativo, axis=1).T
            acumulador_global[tipo].append(df_res_ativo)
            
        except Exception as e:
            print(f"  Erro em {ativo} {tipo}: {e}")

# --- GERAÇÃO DOS GRÁFICOS GLOBAIS ---
for tipo in ["NORMAL", "FUZZY"]:
    if not acumulador_global[tipo]: continue
    
    # Média de todos os ativos para o tipo específico
    df_global = pd.concat(acumulador_global[tipo]).groupby(level=0).mean()
    
    plt.figure(figsize=(12, 7), facecolor='white')
    
    # Seleciona as Top 5 variáveis globais baseadas na média temporal
    top_global = df_global.mean().sort_values(ascending=False).head(5).index
    
    for i, coluna in enumerate(top_global):
        plt.plot(df_global.index, df_global[coluna], 
                 marker='s', 
                 markersize=10,
                 linewidth=3, 
                 label=coluna, 
                 color=CORES_ESTABILIDADE[i % len(CORES_ESTABILIDADE)])

    plt.title(f"ESTABILIDADE TEMPORAL GLOBAL: MODELO {tipo} (Média de 10 Criptoativos)", fontsize=14, fontweight='bold')
    plt.xlabel("Janelas Temporais (Evolução do Teste)", fontsize=11, fontweight='bold')
    plt.ylabel("Impacto SHAP Médio Global", fontsize=11, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10, title="Principais Features")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA_GLOBAL, f"estabilidade_GLOBAL_{tipo}.png"), dpi=300)
    plt.close()

print(f"\n🚀 Sucesso! Gráficos globais salvos em: {PASTA_SAIDA_GLOBAL}")