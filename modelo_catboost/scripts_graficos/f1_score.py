import re
import matplotlib.pyplot as plt
import numpy as np
import os

def gerar_graficos_f1_centro_vertical(caminho_txt):
    if not os.path.exists(caminho_txt):
        print(f"Erro: O arquivo '{caminho_txt}' não foi encontrado.")
        return

    # Lendo o arquivo e ignorando erros de caracteres especiais
    with open(caminho_txt, 'r', encoding='utf-8', errors='ignore') as f:
        conteudo = f.read()

    # Regex atualizada para capturar Ativo e F1-Score Macro
    # Nota: Ajustei o regex para buscar 'F1' seguido de números, que é o padrão comum de logs
    blocos = re.findall(r"RESULTADOS:\s*(\w+).*?F1.*?:\s*([\d\.]+)", conteudo, re.S | re.I)

    if not blocos:
        print("Nenhum dado de F1-Score encontrado no arquivo.")
        return

    dados = []
    for b in blocos:
        nome_completo, f1_val = b
        partes = nome_completo.split('_')
        tipo = 'Fuzzy' if 'FUZZY' in nome_completo else 'Normal'
        # Mantém o Ticker_H (ex: BTC_H30)
        ativo_h = "_".join(partes[1:]) if len(partes) > 1 else partes[0]
        dados.append({'id': ativo_h, 'tipo': tipo, 'f1': float(f1_val)})

    # Organização dos IDs (Ativos_H)
    ids = sorted(list(set([d['id'] for d in dados])))
    x = np.arange(len(ids))
    largura = 0.40

    def add_labels_centro_vertical(rects, ax, casas_decimais=3):
        """Escreve os valores no centro das barras, na VERTICAL e em preto"""
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                posicao_centro = height / 2
                ax.text(rect.get_x() + rect.get_width()/2., posicao_centro,
                        f'{height:.{casas_decimais}f}',
                        ha='center', va='center', color='black', 
                        fontweight='bold', fontsize=7, rotation=90)

    # --- GERANDO O GRÁFICO DE F1-SCORE ---
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    
    f1_n = [next((d['f1'] for d in dados if d['id'] == i and d['tipo'] == 'Normal'), 0) for i in ids]
    f1_f = [next((d['f1'] for d in dados if d['id'] == i and d['tipo'] == 'Fuzzy'), 0) for i in ids]

    # Cores solicitadas: Cinza (#D3D3D3) e Amarelo (#FFFF00)
    barras1_n = ax1.bar(x - largura/2, f1_n, largura, label='Normal', color='#D3D3D3', linewidth=0)
    barras1_f = ax1.bar(x + largura/2, f1_f, largura, label='Fuzzy', color='#FFFF00', linewidth=0)
    
    add_labels_centro_vertical(barras1_n, ax1, 3) # F1 geralmente usa 3 casas decimais
    add_labels_centro_vertical(barras1_f, ax1, 3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ids, rotation=45, ha='right')
    ax1.set_ylabel('F1-Score Macro')
    ax1.set_title('Comparação de Performance: F1-Score Macro (Normal vs. Fuzzificado)')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig("f1_score_vertical.png", dpi=300)
    
    print("Sucesso! Gráfico de F1-Score salvo como 'f1_score_vertical.png'.")
    plt.show()

# EXECUÇÃO
# Certifique-se de que o nome do arquivo .txt está correto
gerar_graficos_f1_centro_vertical('resultados_completos_discretizacao.txt')