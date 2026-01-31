import re
import matplotlib.pyplot as plt
import numpy as np
import os

def gerar_graficos_finais_centro_vertical(caminho_txt):
    if not os.path.exists(caminho_txt):
        print(f"Erro: O arquivo '{caminho_txt}' não foi encontrado.")
        return

    # Lendo o arquivo e ignorando erros de caracteres especiais
    with open(caminho_txt, 'r', encoding='utf-8', errors='ignore') as f:
        conteudo = f.read()

    # Regex para capturar os dados (Ativo, Acurácia e Log Loss)
    blocos = re.findall(r"RESULTADOS:\s*(\w+).*?Acur.*?:\s*([\d\.]+)%.*?Log.*?Loss.*?:\s*([\d\.]+)", conteudo, re.S)

    if not blocos:
        print("Nenhum dado encontrado no arquivo.")
        return

    dados = []
    for b in blocos:
        nome_completo, acuracia, logloss = b
        partes = nome_completo.split('_')
        tipo = 'Fuzzy' if 'FUZZY' in nome_completo else 'Normal'
        ativo_h = "_".join(partes[1:]) if len(partes) > 1 else partes[0]
        dados.append({'id': ativo_h, 'tipo': tipo, 'acc': float(acuracia), 'll': float(logloss)})

    ids = sorted(list(set([d['id'] for d in dados])))
    x = np.arange(len(ids))
    largura = 0.40

    def add_labels_centro_vertical(rects, ax, casas_decimais=2):
        """Escreve os valores no centro das barras, na VERTICAL e em preto"""
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                posicao_centro = height / 2
                ax.text(rect.get_x() + rect.get_width()/2., posicao_centro,
                        f'{height:.{casas_decimais}f}',
                        ha='center', va='center', color='black', 
                        fontweight='bold', fontsize=7, rotation=90) # ROTAÇÃO 90 PARA VERTICAL

    # --- 1. GRÁFICO DE LOG LOSS ---
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ll_n = [next((d['ll'] for d in dados if d['id'] == i and d['tipo'] == 'Normal'), 0) for i in ids]
    ll_f = [next((d['ll'] for d in dados if d['id'] == i and d['tipo'] == 'Fuzzy'), 0) for i in ids]

    # linewidth=0 remove o contorno das barras
    barras1_n = ax1.bar(x - largura/2, ll_n, largura, label='Normal', color='#D3D3D3', linewidth=0)
    barras1_f = ax1.bar(x + largura/2, ll_f, largura, label='Fuzzy', color='#FFFF00', linewidth=0)
    
    add_labels_centro_vertical(barras1_n, ax1, 2)
    add_labels_centro_vertical(barras1_f, ax1, 2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ids, rotation=45, ha='right')
    ax1.set_ylabel('Log Loss')
    ax1.set_title('Consistência dos Modelos - Log Loss')
    ax1.legend()
    plt.tight_layout()
    plt.savefig("log_loss_vertical.png", dpi=300)
    
    # --- 2. GRÁFICO DE ACURÁCIA ---
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    acc_n = [next((d['acc'] for d in dados if d['id'] == i and d['tipo'] == 'Normal'), 0) for i in ids]
    acc_f = [next((d['acc'] for d in dados if d['id'] == i and d['tipo'] == 'Fuzzy'), 0) for i in ids]

    barras2_n = ax2.bar(x - largura/2, acc_n, largura, label='Normal %', color='#D3D3D3', linewidth=0)
    barras2_f = ax2.bar(x + largura/2, acc_f, largura, label='Fuzzy %', color='#FFFF00', linewidth=0)

    add_labels_centro_vertical(barras2_n, ax2, 1)
    add_labels_centro_vertical(barras2_f, ax2, 1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, rotation=45, ha='right')
    ax2.set_ylabel('Acurácia %')
    ax2.set_title('Acurácia %')
    ax2.legend()
    plt.tight_layout()
    plt.savefig("acuracia_vertical.png", dpi=300)

    print("Finalizado! Gráficos com valores verticais e sem bordas salvos com sucesso.")
    plt.show()

# EXECUÇÃO
gerar_graficos_finais_centro_vertical('resultados_completos_discretizacao.txt')