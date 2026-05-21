import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse

from convnext.model import create_mask_rcnn_model, create_image_processor
from convnext.dataset import LunaDataset, create_luna_dataset
import config

def collate_fn(batch):
    return tuple(zip(*batch))

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # O Torchvision MaskRCNN retorna um dicionário de perdas durante o `.train()`
        loss_dict = model(images, targets) 
        
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    # Para métricas de detecção o ideal seria mAP etc, mas para exemplo vamos simplificar
    # Apenas mostrando como capturar output de eval do MaskRCNN
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            # outputs é uma lista de dicionários contendo boxes, labels, masks e scores
            # Implementar COCO metrics ou similar seria o ideal aqui.
    return 0.0 # acc temporariamenre retornando zero

def main(args):
    # Configurar device
    device = torch.device(config.DEVICE)
    print(f"Usando device: {device}")
    
    # Criar processador de imagens e modelo
    image_processor = create_image_processor()
    model = create_mask_rcnn_model(
        num_classes=config.NUM_LABELS
    )
    model.to(device)

    # Carregar datasets com suporte a COCO
    train_ds = create_luna_dataset(
        folder=args.train_folder or config.TRAIN_FOLDER,
        image_processor=image_processor,
        annotation_format=args.annotation_format,
        coco_file=args.train_coco_file
    )
    test_ds = create_luna_dataset(
        folder=args.test_folder or config.TEST_FOLDER,
        image_processor=image_processor,
        annotation_format=args.annotation_format,
        coco_file=args.test_coco_file
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Otimizador
    optimizer = AdamW(model.parameters(), lr=config.LR)
    
    # Loop de treinamento
    best_acc = 0
    for epoch in range(config.NUM_EPOCHS):
        # Treinar
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: Loss médio = {avg_loss:.4f}")
        
        # Avaliar
        acc = evaluate(model, test_loader, device)
        print(f"Validação: Acurácia = {acc*100:.2f}%")
        
        # Salvar melhor modelo
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
            }, os.path.join(config.SAVE_MODEL_PATH, 'best_model.pth'))
            print(f"Novo melhor modelo salvo com acurácia: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar modelo ConvNeXtV2')
    parser.add_argument('--train_folder', type=str, help='Pasta com dados de treino')
    parser.add_argument('--test_folder', type=str, help='Pasta com dados de teste')
    parser.add_argument('--pretrained', action='store_true', default=True, 
                       help='Usar modelo pré-treinado')
    
    # Argumentos para anotações COCO
    parser.add_argument('--annotation_format', type=str, default='auto', 
                       choices=['auto', 'txt', 'coco'],
                       help='Formato das anotações: auto, txt ou coco')
    parser.add_argument('--train_coco_file', type=str, 
                       help='Arquivo COCO para dados de treino')
    parser.add_argument('--test_coco_file', type=str,
                       help='Arquivo COCO para dados de teste')
    
    args = parser.parse_args()
    main(args)