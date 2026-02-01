import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# --- CONFIGURAÇÕES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "models")
PASTA_DADOS_RAIZ = os.path.join(SCRIPT_DIR, "dados_normalizados")
PASTA_SAIDA_GLOBAL = os.path.join(SCRIPT_DIR, "shap_analise_global_consolidada")
N_AMOSTRAS = 500
COR_AMARELO_PREMIUM = "#FFD700"  # Ouro das matrizes
os.makedirs(PASTA_SAIDA_GLOBAL, exist_ok=True)

def carregar_dados(tipo, ativo):
    sub = "normal" if tipo == "NORMAL" else "fuzzy"
    path = os.path.join(PASTA_DADOS_RAIZ, sub, "test")
    if not os.path.exists(path): return None
    for f in os.listdir(path):
        if ativo.upper() in f.upper() and f.endswith(".csv"):
            return pd.read_csv(os.path.join(path, f)).select_dtypes(include=[np.number])
    return None

# --- ACUMULADORES GLOBAIS ---
# Armazenaremos as séries de importância de cada modelo para tirar a média final
importancias_globais = {"NORMAL": [], "FUZZY": []}

# --- PROCESSAMENTO ---
arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.endswith('.cbm')]
pares = {}
for f in arquivos:
    p = f.replace(".cbm", "").split("_")
    if len(p) >= 4:
        chave = (p[2], p[3])
        if chave not in pares: pares[chave] = {}
        pares[chave][p[1]] = f

print(f"Iniciando Extração SHAP Global de {len(pares)} configurações...")

for (ativo, horiz), mods in pares.items():
    for tipo in ["NORMAL", "FUZZY"]:
        try:
            if tipo not in mods: continue
            df_raw = carregar_dados(tipo, ativo)
            if df_raw is None: continue
            
            model = CatBoostClassifier().load_model(os.path.join(PASTA_MODELOS, mods[tipo]))
            cols = model.feature_names_
            X = df_raw[cols].sample(n=min(len(df_raw), N_AMOSTRAS), random_state=42)
            
            sv = model.get_feature_importance(Pool(X), type='ShapValues', thread_count=-1)
            
            if len(sv.shape) == 3:
                val = np.abs(sv[:, :, :-1]).mean(axis=(0, 1))
            else:
                val = np.abs(sv[:, :-1]).mean(axis=0)
                
            ser = pd.Series(val, index=cols)

            if tipo == "FUZZY":
                # Agrupamento por indicador (removendo o sufixo da classe fuzzy)
                ser = ser.groupby(lambda x: x.split('_')[0]).sum()
            
            importancias_globais[tipo].append(ser)
            
        except Exception as e:
            print(f"  Erro no acúmulo de {ativo} {tipo}: {e}")

# --- GERAÇÃO DOS GRÁFICOS GLOBAIS ---
for tipo in ["NORMAL", "FUZZY"]:
    if not importancias_globais[tipo]: continue
    
    # Consolida todos os ativos tirando a média da importância SHAP
    df_consolidado = pd.concat(importancias_globais[tipo], axis=1).mean(axis=1).sort_values(ascending=True)

    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Mostra os 15 indicadores mais influentes no projeto inteiro
    top_15 = df_consolidado.tail(15)
    top_15.plot(kind='barh', color=COR_AMARELO_PREMIUM, edgecolor='#B8860B', linewidth=1.5)

    plt.title(f"IMPORTÂNCIA SHAP GLOBAL: MODELO {tipo}\n(Consolidado de 10 Criptoativos)", fontsize=14, fontweight='bold')
    plt.xlabel("Impacto Médio Absoluto Global (mean|SHAP value|)", fontsize=12, fontweight='bold')
    plt.ylabel("Indicadores / Variáveis Agrupadas", fontsize=12, fontweight='bold')

    # Adicionando rótulos de valores
    for i, v in enumerate(top_15):
        plt.text(v + (v*0.005), i, f'{v:.4f}', color='black', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA_GLOBAL, f"SHAP_GLOBAL_{tipo}.png"), dpi=300)
    plt.close()
    print(f" Gráfico Global {tipo} gerado com sucesso.")

print(f"\n Análise Global Finalizada em: {PASTA_SAIDA_GLOBAL}")