import matplotlib.pyplot as plt
from PIL import Image
import os

def criar_painel_inferencias():
    
    caminhos = [
        "resultados_inferencia_com_gt_15mm/pred_img_1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405_139.jpg",      # Imagem 1: > 15mm
        "resultados_inferencia_com_gt_10-15mm/pred_img_1.3.6.1.4.1.14519.5.2.1.6279.6001.131939324905446238286154504249_84.jpg",   # Imagem 2: 10 a 15mm
        "resultados_inferencia_com_gt_6-10mm/pred_img_1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405_265.jpg",    # Imagem 3: 6 a 10mm
        "resultados_inferencia_com_gt_6mm/pred_img_1.3.6.1.4.1.14519.5.2.1.6279.6001.137375498893536422914241295628_89.jpg"        # Imagem 4: <= 6mm
    ]

    titulos = [
        "(a) Nódulo com $d > 15$mm",
        "(b) Nódulo entre $10$mm e $15$mm",
        "(c) Nódulo entre $5$mm e $10$mm",
        "(d) Nódulo com $d \leq 5$mm"
    ]

    # Cria o grid 2x2. figsize=(12, 12) garante uma imagem final grande e nítida
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Achata a matriz de eixos para facilitar o loop (de 2x2 para uma lista de 4)
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        
        # Verifica se você colocou o caminho da imagem certinho
        if os.path.exists(caminhos[i]):
            img = Image.open(caminhos[i])
            ax.imshow(img)
        else:
            # Caso não ache a imagem, deixa um aviso vermelho na tela
            ax.text(0.5, 0.5, f"Imagem não encontrada:\n{caminhos[i]}", 
                    color='red', ha='center', va='center', fontsize=12)
            ax.set_facecolor('#f0f0f0')

        # Configura o título de cada sub-imagem
        ax.set_title(titulos[i], fontsize=16, pad=10)
        
        # Remove os eixos (aqueles números indesejados de X e Y do lado da foto)
        ax.axis('off')

    # Ajusta o espaçamento para não ficar nada encavalado
    plt.tight_layout()

    # Salva a imagem final com 300 DPI (Resolução exigida por revistas científicas e bancas)
    nome_saida = "painel_inferencias_tcc.png"
    plt.savefig(nome_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSucesso! O painel foi gerado e salvo como: '{nome_saida}'")

if __name__ == "__main__":
    print("Montando o painel de imagens...")
    criar_painel_inferencias()