import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# --- CONFIGURAÇÕES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "models")
PASTA_DADOS_RAIZ = os.path.join(SCRIPT_DIR, "dados_normalizados")
PASTA_SAIDA = os.path.join(SCRIPT_DIR, "shap_analise_global")
N_AMOSTRAS = 500
COR_AMARELO_PREMIUM = "#FFD700"  # Ouro das matrizes
os.makedirs(PASTA_SAIDA, exist_ok=True)

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
    print(f"📊 Analisando SHAP Global: {ativo} | {horiz}")
    pasta_ativo = os.path.join(PASTA_SAIDA, f"{ativo}_{horiz}")
    os.makedirs(pasta_ativo, exist_ok=True)

    for tipo in ["NORMAL", "FUZZY"]:
        try:
            if tipo not in mods: continue
            df_raw = carregar_dados(tipo, ativo)
            if df_raw is None: continue
            
            model = CatBoostClassifier().load_model(os.path.join(PASTA_MODELOS, mods[tipo]))
            cols = model.feature_names_
            X = df_raw[cols].sample(n=min(len(df_raw), N_AMOSTRAS), random_state=42)
            
            # Cálculo de SHAP
            sv = model.get_feature_importance(Pool(X), type='ShapValues', thread_count=-1)
            
            # Remove a última coluna (bias) e tira a média absoluta
            if len(sv.shape) == 3:
                val = np.abs(sv[:, :, :-1]).mean(axis=(0, 1))
            else:
                val = np.abs(sv[:, :-1]).mean(axis=0)
                
            ser = pd.Series(val, index=cols)

            plt.figure(figsize=(11, 7), facecolor='white') # Fundo branco limpo
            
            if tipo == "FUZZY":
                # Agrupamento Econômico
                ser = ser.groupby(lambda x: x.split('_')[0]).sum().sort_values(ascending=True)
                plt.title(f"Importância das Features ({tipo}) - {ativo} {horiz}", fontsize=13, fontweight='bold')
            else:
                ser = ser.sort_values(ascending=True)
                plt.title(f"Importância das Features ({tipo}) - {ativo} {horiz}", fontsize=13, fontweight='bold')

            # Plotagem com o AMARELO DAS MATRIZES
            # Usando tail(15) para mostrar os indicadores mais importantes no topo
            plot = ser.tail(15).plot(kind='barh', color=COR_AMARELO_PREMIUM, edgecolor='#B8860B', linewidth=1)

            # --- ESTÉTICA E NOMES NOS EIXOS ---
            plt.xlabel("Impacto Médio (mean|SHAP value|)", fontsize=11, fontweight='bold', labelpad=10)
            plt.ylabel("Indicadores Econômicos / Classes", fontsize=11, fontweight='bold', labelpad=10)
            
            # Adicionando os valores nas pontas das barras para facilitar leitura
            for i, v in enumerate(ser.tail(15)):
                plt.text(v + (v*0.01), i, f'{v:.4f}', color='black', va='center', fontsize=9)

            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_ativo, f"importancia_{tipo}.png"), dpi=150)
            plt.close()
            print(f"  ✅ {tipo} concluído.")
            
        except Exception as e:
            print(f"  ❌ Erro em {ativo} {horiz} {tipo}: {e}")

print(f"\n✨ Tudo pronto! Gráficos em amarelo ouro salvos em: {PASTA_SAIDA}")