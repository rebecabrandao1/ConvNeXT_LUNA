import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config 
from convnext.model import create_mask_rcnn_model, create_image_processor
from convnext.dataset import create_luna_dataset # Certifique-se de que esta função existe no seu dataset.py

def collate_fn(batch):
    """
    Necessário para desempacotar batches de tamanhos variáveis (detecção).
    """
    return tuple(zip(*batch))

def get_optimizer(model, lr=1e-4):
    # AdamW é o padrão para ConvNeXt V2
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
 
    pbar = tqdm(data_loader, desc="Treinando")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    return total_loss / len(data_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando treino no dispositivo: {device}")

    # 1. Preparar os Datasets e Loaders
    image_processor = create_image_processor()
    

    train_ds = create_luna_dataset(folder=config.TRAIN_FOLDER, image_processor=image_processor)
    test_ds = create_luna_dataset(folder=config.TEST_FOLDER, image_processor=image_processor)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # 2. Inicializar Modelo e Otimizador
    model = create_mask_rcnn_model(num_classes=config.NUM_LABELS)
    model.to(device)
    
    optimizer = get_optimizer(model, lr=config.LR)
    
    os.makedirs(config.SAVE_MODEL_PATH, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Época {epoch+1}/{config.NUM_EPOCHS} ---")
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Loss Médio da Época: {loss:.4f}")
       
        save_path = os.path.join(config.SAVE_MODEL_PATH, f"detector_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Modelo salvo em: {save_path}")
        print(f"Boxes: {targets[0]['boxes']}")
        print(f"Labels: {targets[0]['labels']}")

if __name__ == "__main__":
    main()