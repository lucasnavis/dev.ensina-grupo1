import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def gerar_matrizes_amarelo_premium(caminho_txt):
    # 1. Preparação da Pasta
    pasta_destino = 'matriz_confusão'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    if not os.path.exists(caminho_txt):
        print(f"Erro: Arquivo {caminho_txt} não encontrado.")
        return

    # 2. Leitura e Limpeza (Tratando caracteres especiais do log)
    with open(caminho_txt, 'r', encoding='utf-8', errors='ignore') as f:
        texto_sujo = f.read()
    
    texto_limpo = "".join(i for i in texto_sujo if ord(i) < 128)

    # 3. Extração dos Dados
    padrao = r"RESULTADOS:\s*(NORMAL|FUZZY)_(.*?)_(H\d+).*?\[\[(.*?)]]"
    blocos = re.findall(padrao, texto_limpo, re.S)

    comparativo = {}
    for b in blocos:
        tipo = b[0].upper()
        ativo = b[1].strip().upper()
        horiz = b[2].strip().upper()
        chave = f"{ativo}_{horiz}"
        
        nums = re.findall(r"\d+", b[3])
        if len(nums) == 9:
            matriz = np.array([int(n) for n in nums]).reshape(3, 3)
            if chave not in comparativo:
                comparativo[chave] = {}
            comparativo[chave][tipo] = matriz

    # 4. PALETA DE CORES: Amarelo Premium (Dourado/Âmbar)
    # Vai de um creme muito claro até um dourado profundo, sem cinza.
    cores_amarelas = ["#FFFBEE", "#FFD700", "#B8860B"] 
    cmap_premium = LinearSegmentedColormap.from_list("gold", cores_amarelas)

    labels_eixo = ['0', '1', '2']
    sucessos = 0

    print(f"Processando {len(comparativo)} pares de ativos...")

    for chave in sorted(comparativo.keys()):
        modelos = comparativo[chave]
        
        if 'NORMAL' in modelos and 'FUZZY' in modelos:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Matriz NORMAL
            sns.heatmap(modelos['NORMAL'], annot=True, fmt='d', cmap=cmap_premium, ax=ax1, cbar=False,
                        xticklabels=labels_eixo, yticklabels=labels_eixo,
                        annot_kws={"color": "black", "fontweight": "bold", "rotation": 0, "size": 13})
            ax1.set_title(f"NORMAL - {chave}", fontweight='bold', fontsize=12)
            ax1.set_xlabel('Previsão')
            ax1.set_ylabel('Real')

            # Matriz FUZZY
            sns.heatmap(modelos['FUZZY'], annot=True, fmt='d', cmap=cmap_premium, ax=ax2, cbar=False,
                        xticklabels=labels_eixo, yticklabels=labels_eixo,
                        annot_kws={"color": "black", "fontweight": "bold", "rotation": 0, "size": 13})
            ax2.set_title(f"FUZZY - {chave}", fontweight='bold', fontsize=12)
            ax2.set_xlabel('Previsão')
            ax2.set_ylabel('Real')

            plt.tight_layout()
            
            # Salvando
            nome_arq = os.path.join(pasta_destino, f"comparativo_{chave}.png")
            plt.savefig(nome_arq, dpi=150)
            plt.close()
            sucessos += 1
            print(f"✅ {sucessos} - {chave} gerado com sucesso.")

    print(f"\n🚀 Pronto! {sucessos} comparativos premium salvos na pasta '{pasta_destino}'.")

# EXECUÇÃO
gerar_matrizes_amarelo_premium('resultados_completos_discretizacao.txt')