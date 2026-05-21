import torch
import os
from convnext.dataset import create_luna_dataset
from convnext.model import create_image_processor

def test_dataset():
    print('Iniciando teste do dataset LUNA16 adaptado para Mask R-CNN...')
    processor = create_image_processor()
    
    test_folder = 'dataset/dataset_10-15mm_train'
    if not os.path.exists(test_folder):
        subdirs = [f for f in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', f))]
        if subdirs:
            test_folder = os.path.join('dataset', subdirs[0])
            
    print(f'Carregando dados da pasta: {test_folder}')
    
    # Precisamos encontrar o JSON coco correspondente para a pasta se existir
    coco_file = test_folder + '.json'
    if not os.path.exists(coco_file) and test_folder.endswith('_train'):
        possible_json = test_folder.replace('_train', '') + '_train.json'
        if os.path.exists(possible_json):
            coco_file = possible_json

    # Deixar em null se não achar
    if not os.path.exists(coco_file):
        coco_file = None

    ds = create_luna_dataset(test_folder, processor, annotation_format='auto', coco_file=coco_file)
    
    print(f'Total de imagens carregadas: {len(ds)}')
    
    if len(ds) == 0:
        return
        
    for i in range(min(5, len(ds))):
        img, tgt = ds[i]
        print(f'\n--- Amostra {i} ---')
        print(f'Shape Imagem Tensor: {img.shape}')
        print(f'Classe Label: {tgt["labels"]}')
        print(f'Box (Coordenadas [x1, y1, x2, y2]): {tgt["boxes"]}')
        if "masks" in tgt:
            print(f'Mask Tensor Shape: {tgt["masks"].shape}')
            if tgt["masks"].numel() > 0:
                unique_vals = torch.unique(tgt['masks'])
                print(f'Valores unicos na mascara (deve ser 0 e 1): {unique_vals}')
            
if __name__ == "__main__":
    test_dataset()