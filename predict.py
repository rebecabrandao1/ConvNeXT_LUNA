import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import config
from convnext.model import create_mask_rcnn_model, create_image_processor

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

def predict_and_visualize(model, image_path, image_processor, device, save_path, threshold=0.1):
    """Faz a predição e salva a imagem na pasta de resultados."""
    img = Image.open(image_path).convert("RGB")
    processed = image_processor(img, return_tensors="pt")
    image_tensor = processed["pixel_values"].squeeze(0).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # --- RAIO-X DO MODELO ---
    print("\n[DEBUG] O que o modelo cuspiu:")
    print(f"Total de caixas iniciais: {len(prediction['boxes'])}")
    if len(prediction['scores']) > 0:
        print(f"Score máximo: {prediction['scores'].max().item():.4f}")
    # ------------------------

    scores = prediction['scores'].cpu().numpy()
        

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
            
            # --- Desenhar Máscara ---
            binary_mask = mask > 0.5
            mask_layer = np.zeros((*binary_mask.shape, 4))
            mask_layer[binary_mask] = [1, 0, 0, 0.4] 
            ax.imshow(mask_layer)

    ax.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # CRUCIAL: Libera a memória após salvar!

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    PESO_TREINADO = 'outputs/detector_epoch_400.pth' 
    PASTA_TESTE = "dataset/dataset_10-15mm_test"
    
    # Cria uma pasta nova para não misturar os resultados
    PASTA_RESULTADOS = "resultados_inferencia"
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    
    try:
        model = load_model(PESO_TREINADO, device)
        image_processor = create_image_processor()
        
        # Pega a lista de todas as imagens .jpg na pasta
        imagens = glob.glob(os.path.join(PASTA_TESTE, "*.jpg"))
        total_imagens = len(imagens)
        
        print(f"\nIniciando inferência em {total_imagens} imagens...")
        print(f"Os resultados serão salvos na pasta: {PASTA_RESULTADOS}/")
        print("-" * 50)
        
        for idx, img_path in enumerate(imagens, 1):
            nome_arquivo = os.path.basename(img_path)
            caminho_salvar = os.path.join(PASTA_RESULTADOS, f"pred_{nome_arquivo}")
            
            print(f"[{idx}/{total_imagens}] Processando: {nome_arquivo}")
            predict_and_visualize(model, img_path, image_processor, device, caminho_salvar, threshold=0.1)
            
        print("\n=== Concluído! Todas as imagens foram processadas. ===")
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo ou pasta não encontrado. Verifique os caminhos.\n{e}")

if __name__ == "__main__":
    main()