import torch
import numpy as np
from PIL import Image
import argparse
import os

from convnext.model import create_model, create_image_processor
import config

def predict_image(model, image_processor, img_path, device):
    """
    Faz predição em uma única imagem.
    """
    img = Image.open(img_path).convert("L")
    img_np = np.expand_dims(np.array(img), axis=-1)
    inputs = image_processor([img_np], return_tensors="pt", image_mean=[0.5], image_std=[0.5])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
    
    return pred, confidence

def main(args):
    device = torch.device(config.DEVICE)
    
    # Carregar modelo
    model = create_model(
        num_classes=config.NUM_LABELS,
        pretrained=False,
        num_channels=config.NUM_CHANNELS
    )
    
    # Carregar checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Criar processador
    image_processor = create_image_processor()
    
    # Fazer predição
    pred, confidence = predict_image(model, image_processor, args.image_path, device)
    
    print(f"Predição para {args.image_path}:")
    print(f"Classe: {pred}")
    print(f"Confiança: {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fazer inferência com modelo treinado')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Caminho para o modelo salvo')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Caminho para a imagem')
    
    args = parser.parse_args()
    main(args)