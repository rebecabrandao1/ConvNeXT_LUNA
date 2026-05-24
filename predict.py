import os
import glob
import torch
import numpy as np
import matplotlib.subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F

import config
from convnext.model import create_mask_rcnn_model

def load_model(weights_path, device):
    print(f"Carregando pesos de: {weights_path}")
    model = create_mask_rcnn_model(num_classes=config.NUM_LABELS)
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() 
    return model

# --- FUNÇÃO NOVA: LER O GROUND TRUTH ---
def ler_ground_truth_yolo(txt_path, img_width, img_height):
    """Lê as anotações originais em formato YOLO (.txt) e converte para pixels."""
    gt_boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) >= 5:
                    # Formato YOLO: classe x_centro y_centro largura altura
                    _, x_c, y_c, w, h = map(float, partes[:5])
                    
                    # Desfaz a normalização multiplicando pelo tamanho real da imagem
                    largura = w * img_width
                    altura = h * img_height
                    x_centro = x_c * img_width
                    y_centro = y_c * img_height
                    
                    xmin = x_centro - (largura / 2)
                    ymin = y_centro - (altura / 2)
                    xmax = x_centro + (largura / 2)
                    ymax = y_centro + (altura / 2)
                    
                    gt_boxes.append([xmin, ymin, xmax, ymax])
    return gt_boxes

def predict_and_visualize(model, image_path, txt_path, device, save_path, threshold=0.8):
    """Faz a predição e desenha GT (azul) e Modelo (vermelho)."""
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    image_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    scores = prediction['scores'].cpu().numpy()
    keep = scores > threshold
    
    boxes = prediction['boxes'][keep].cpu().numpy()
    masks = prediction['masks'][keep].cpu().numpy()
    scores = scores[keep]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    # --- 1. DESENHAR O GROUND TRUTH (EM AZUL) ---
    gt_boxes = ler_ground_truth_yolo(txt_path, img_width, img_height)
    for gt_box in gt_boxes:
        xmin, ymin, xmax, ymax = gt_box
        # Retângulo azul tracejado para diferenciar bem
        rect_gt = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                    linewidth=3, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect_gt)
        ax.text(xmin, ymin - 8, "GT", color='blue', fontsize=12, weight='bold')

    # --- 2. DESENHAR AS PREDIÇÕES DO MODELO (EM VERMELHO) ---
    if len(boxes) == 0:
        pass # Mantém apenas a imagem com o GT se o modelo não achar nada
    else:
        for i in range(len(boxes)):
            box = boxes[i]
            mask = masks[i, 0] 
            score = scores[i]
            
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Movimentei a porcentagem para baixo (ymax) para não ficar em cima do letreiro "GT"
            ax.text(xmin, ymax + 15, f"{score*100:.1f}%", 
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Contorno da máscara
            binary_mask = mask > 0.5
            ax.contour(binary_mask, colors='red', linewidths=1.5, levels=[0.5])

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig) 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # ⚠️ MUDE AQUI PARA O PESO CORRETO SE ESTIVER USANDO O DA ÉPOCA 400!
    PESO_TREINADO = 'outputs/detector_epoch_38.pth' 
    
    pastas_teste = {
        "10-15mm": "dataset/dataset_10-15mm_test",
        "15mm": "dataset/dataset_15mm_test",
        "6-10mm": "dataset/dataset_6-10mm_test",
        "6mm": "dataset/dataset_6mm_test"
    }
    
    try:
        model = load_model(PESO_TREINADO, device)
        
        for nome_teste, pasta_origem in pastas_teste.items():
            print(f"\n\n{'='*50}")
            print(f" INICIANDO INFERÊNCIA: {nome_teste}")
            print(f"{'='*50}")
            
            if not os.path.exists(pasta_origem):
                print(f"Aviso: A pasta '{pasta_origem}' não foi encontrada. Pulando...")
                continue

            # Nome da pasta final ajustado para sabermos que tem o GT desenhado
            pasta_resultados = f"resultados_inferencia_com_gt_{nome_teste}"
            os.makedirs(pasta_resultados, exist_ok=True)
            
            imagens = glob.glob(os.path.join(pasta_origem, "*.jpg"))
            total_imagens = len(imagens)
            
            print(f"Encontradas {total_imagens} imagens. Salvando em: {pasta_resultados}/")
            
            for idx, img_path in enumerate(imagens, 1):
                nome_arquivo = os.path.basename(img_path)
                caminho_salvar = os.path.join(pasta_resultados, f"pred_{nome_arquivo}")
                
                # --- O SEGREDO ESTÁ AQUI ---
                # Pega o mesmo nome da imagem, mas troca .jpg por .txt para achar a anotação
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                
                print(f"[{idx}/{total_imagens}] Processando: {nome_arquivo}")
                
                predict_and_visualize(model, img_path, txt_path, device, caminho_salvar, threshold=0.8)
                
        print("\n=== Concluído! Todas as inferências foram processadas. ===")
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo ou pasta não encontrado. Verifique os caminhos.\n{e}")

if __name__ == "__main__":
    main()