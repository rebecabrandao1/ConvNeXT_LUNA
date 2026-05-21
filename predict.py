import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import config
from convnext.model import create_mask_rcnn_model, create_image_processor

def load_model(weights_path, device):
    """Carrega o modelo Mask R-CNN com os pesos treinados."""
    print(f"Carregando pesos de: {weights_path}")
    model = create_mask_rcnn_model(num_classes=config.NUM_LABELS)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() 
    return model

def predict_and_visualize(model, image_path, image_processor, device, threshold=0.5):
    """Faz a predição em uma imagem e plota o resultado."""
    img = Image.open(image_path).convert("RGB")
    processed = image_processor(img, return_tensors="pt")
    image_tensor = processed["pixel_values"].to(device)

    print(f"Analisando imagem: {image_path}...")
    with torch.no_grad():
 
        prediction = model(image_tensor)[0] 

    scores = prediction['scores'].cpu().numpy()
    keep = scores > threshold
    
    boxes = prediction['boxes'][keep].cpu().numpy()
    masks = prediction['masks'][keep].cpu().numpy()
    labels = prediction['labels'][keep].cpu().numpy()
    scores = scores[keep]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    if len(boxes) == 0:
        print("Nenhum nódulo detectado com a confiança mínima.")
        ax.set_title("Nenhum Nódulo Detectado", color='green', fontsize=14)
    else:
        print(f"Encontrado(s) {len(boxes)} nódulo(s) suspeito(s)!")
        
        for i in range(len(boxes)):
            box = boxes[i]
            mask = masks[i, 0] 
            score = scores[i]
            
            # --- Desenhar Bounding Box ---
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Texto com a Confiança (ex: 92.5%)
            ax.text(xmin, ymin - 5, f"{score*100:.1f}%", 
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # --- Desenhar Máscara (Segmentação) ---
            # Binarizar a máscara (o modelo cospe probabilidades, queremos 0 ou 1)
            binary_mask = mask > 0.5
            
            # Criar uma camada vermelha transparente onde a máscara existe
            mask_layer = np.zeros((*binary_mask.shape, 4))
            mask_layer[binary_mask] = [1, 0, 0, 0.4] # RGBA (Vermelho com 40% de opacidade)
            ax.imshow(mask_layer)

    ax.axis('off') # Esconde os eixos para ficar bonito no TCC
    plt.tight_layout()
    
    # Salvar o resultado
    output_filename = "resultado_predicao.png"
    plt.savefig(output_filename, dpi=300) # dpi=300 é o padrão ouro para artigos/TCC
    print(f"Imagem salva como {output_filename}")
    #plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    PESO_TREINADO = 'outputs/detector_epoch_400.pth' 
 
    IMAGEM_TESTE = "dataset/dataset_10-15mm_test/img_1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375_184.jpg"
    
    try:
        model = load_model(PESO_TREINADO, device)
        image_processor = create_image_processor()
        
        predict_and_visualize(model, IMAGEM_TESTE, image_processor, device, threshold=0.5)
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique os caminhos.\n{e}")

if __name__ == "__main__":
    main()