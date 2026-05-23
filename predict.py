import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F # <-- ADICIONADO AQUI

import config
from convnext.model import create_mask_rcnn_model

def load_model(weights_path, device):
    print(f"Carregando pesos de: {weights_path}")
    # Nota: Não precisamos mais do create_image_processor aqui
    model = create_mask_rcnn_model(num_classes=config.NUM_LABELS)
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() 
    return model

def predict_and_visualize(model, image_path, device, save_path, threshold=0.8):
    """Faz a predição e salva a imagem na pasta de resultados."""
    img = Image.open(image_path).convert("RGB")
    
    # --- A CIRURGIA FOI AQUI ---
    # Usamos o método nativo do PyTorch que mantém os valores entre 0.0 e 1.0,
    # exatamente como o seu test_mask_dataset.py faz no treino!
    image_tensor = F.to_tensor(img).to(device)
    # ---------------------------

    with torch.no_grad():
        # Mask R-CNN espera uma lista de tensores
        prediction = model([image_tensor])[0]

    # --- RAIO-X DO MODELO ---
    print("\n[DEBUG] O que o modelo cuspiu:")
    print(f"Total de caixas iniciais: {len(prediction['boxes'])}")
    if len(prediction['scores']) > 0:
        print(f"Score máximo: {prediction['scores'].max().item():.4f}")
    # ------------------------

    scores = prediction['scores'].cpu().numpy()
    keep = scores > threshold
    
    boxes = prediction['boxes'][keep].cpu().numpy()
    masks = prediction['masks'][keep].cpu().numpy()
    scores = scores[keep]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    # Se não achou nada, apenas salva a imagem original e fecha
    if len(boxes) == 0:
        ax.set_title("Nenhum Nódulo Detectado", color='green', fontsize=14)
    else:
        for i in range(len(boxes)):
            box = boxes[i]
            mask = masks[i, 0] 
            score = scores[i]
            
            # --- Desenhar Bounding Box ---
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(xmin, ymin - 5, f"{score*100:.1f}%", 
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # --- Desenhar Máscara (Apenas Contorno) ---
            binary_mask = mask > 0.5
            
            # O ax.contour desenha uma linha exatamente na divisa entre o 0 e o 1
            ax.contour(binary_mask, colors='red', linewidths=1.5, levels=[0.5])

    ax.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig) 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    PESO_TREINADO = 'outputs/detector_epoch_38.pth' 
    
    # --- LISTA DE PASTAS PARA INFERÊNCIA ---
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

            pasta_resultados = f"resultados_inferencia_{nome_teste}"
            os.makedirs(pasta_resultados, exist_ok=True)
            
            imagens = glob.glob(os.path.join(pasta_origem, "*.jpg"))
            total_imagens = len(imagens)
            
            print(f"Encontradas {total_imagens} imagens. Salvando em: {pasta_resultados}/")
            
            for idx, img_path in enumerate(imagens, 1):
                nome_arquivo = os.path.basename(img_path)
                caminho_salvar = os.path.join(pasta_resultados, f"pred_{nome_arquivo}")
                
                print(f"[{idx}/{total_imagens}] Processando: {nome_arquivo}")
                
                predict_and_visualize(model, img_path, device, caminho_salvar, threshold=0.8)
                
        print("\n=== Concluído! Todas as inferências foram processadas e separadas por pasta. ===")
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo ou pasta não encontrado. Verifique os caminhos.\n{e}")

if __name__ == "__main__":
    main()