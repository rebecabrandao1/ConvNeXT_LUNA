import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class LunaDataset(Dataset):
    def __init__(self, folder, image_processor, transform=None, annotation_format='txt', coco_file=None):
        """
        Dataset para imagens médicas do LUNA16 com suporte a múltiplos formatos de anotação.
        
        Args:
            folder: Pasta com as imagens
            image_processor: Processador de imagens
            transform: Transformações opcionais
            annotation_format: 'txt', 'coco', ou 'auto'
            coco_file: Caminho para arquivo COCO JSON (se annotation_format='coco')
        """
        self.samples = []
        self.image_processor = image_processor
        self.transform = transform
        self.annotation_format = annotation_format
        self.folder = folder
        
        # Carregar anotações baseado no formato
        if annotation_format == 'coco' or (annotation_format == 'auto' and coco_file):
            self._load_coco_annotations(coco_file or self._find_coco_file())
        elif annotation_format == 'txt' or annotation_format == 'auto':
            self._load_txt_annotations()
        else:
            raise ValueError(f"Formato de anotação não suportado: {annotation_format}")
        
        print(f"Encontradas {len(self.samples)} imagens em {folder} (formato: {self.annotation_format})")
    
    def _find_coco_file(self):
        """Procura por arquivos COCO JSON na pasta."""
        possible_names = ['annotations.json', 'coco.json', 'labels.json', 'train.json', 'test.json']
        
        for name in possible_names:
            coco_path = os.path.join(self.folder, name)
            if os.path.exists(coco_path):
                return coco_path
        
        # Procurar em pasta pai
        parent_folder = os.path.dirname(self.folder)
        for name in possible_names:
            coco_path = os.path.join(parent_folder, name)
            if os.path.exists(coco_path):
                return coco_path
        
        return None
    
    def _load_coco_annotations(self, coco_file):
        """Carrega anotações no formato COCO."""
        if not coco_file or not os.path.exists(coco_file):
            print(f"Arquivo COCO não encontrado: {coco_file}")
            self.annotation_format = 'txt'  # Fallback para txt
            self._load_txt_annotations()
            return
        
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Organizar anotações por imagem
            annotations_by_image = defaultdict(list)
            for ann in coco_data.get('annotations', []):
                annotations_by_image[ann['image_id']].append(ann)
            
            # Criar mapeamento de IDs para arquivos
            image_info = {img['id']: img for img in coco_data.get('images', [])}
            
            # Mapear categorias
            categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            
            # Processar cada imagem
            for img_id, img_info in image_info.items():
                img_filename = img_info['file_name']
                img_path = os.path.join(self.folder, img_filename)
                
                if os.path.exists(img_path):
                    # Pegar anotações desta imagem
                    img_annotations = annotations_by_image.get(img_id, [])
                    
                    if img_annotations:
                        # Para classificação, usar a primeira categoria encontrada
                        # Para detecção, poderia ser adaptado para manter todas
                        first_ann = img_annotations[0]
                        category_id = first_ann['category_id']
                        
                        # Converter category_id para label (0, 1, 2, ...)
                        # Assumindo que category_ids são sequenciais ou criar mapeamento
                        sorted_categories = sorted(categories.keys())
                        label = sorted_categories.index(category_id) if category_id in sorted_categories else 0
                        
                        self.samples.append({
                            'image_path': img_path,
                            'label': label,
                            'category_name': categories.get(category_id, 'unknown'),
                            'annotations': img_annotations,
                            'image_info': img_info
                        })
                    else:
                        # Imagem sem anotações - assumir classe 0 (sem nódulo)
                        self.samples.append({
                            'image_path': img_path,
                            'label': 0,
                            'category_name': 'background',
                            'annotations': [],
                            'image_info': img_info
                        })
            
            print(f" Carregadas {len(self.samples)} imagens do COCO")
            print(f"Categorias encontradas: {list(categories.values())}")
            
        except Exception as e:
            print(f" Erro ao carregar COCO: {e}")
            print("Tentando formato TXT...")
            self.annotation_format = 'txt'
            self._load_txt_annotations()
    
    def _load_txt_annotations(self):
        """Carrega anotações no formato TXT (original)."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        for fname in os.listdir(self.folder):
            if any(fname.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(self.folder, fname)
                base_name = os.path.splitext(fname)[0]
                txt_path = os.path.join(self.folder, f"{base_name}.txt")
                
                if os.path.exists(txt_path):
                    try:
                        # ✅ pega tamanho real
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size

                        img_annotations = []
                        with open(txt_path) as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    label = int(parts[0])
                                    if label == 0:
                                        label = 1  # 0 é background no MaskRCNN

                                    x_c, y_c, w, h = map(float, parts[1:])
                                    
                                    abs_x = (x_c - w/2) * img_w
                                    abs_y = (y_c - h/2) * img_h
                                    abs_w = w * img_w
                                    abs_h = h * img_h
                                    
                                    img_annotations.append({
                                        'bbox': [abs_x, abs_y, abs_w, abs_h],
                                        'category_id': label
                                    })
                    
                        self.samples.append({
                            'image_path': img_path,
                            'label': label if img_annotations else 0,
                            'category_name': f'class_{label}' if img_annotations else 'background',
                            'annotations': img_annotations,
                            'image_info': {'file_name': fname}
                        })
                    except Exception as e:
                        print(f" Erro ao processar {txt_path}: {e}")
        
        print(f" Carregadas {len(self.samples)} imagens do formato TXT")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        
        # Carregar imagem
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        
        # Aplicamos as transformações
        processed = self.image_processor(img, return_tensors="pt")
        image_tensor = processed["pixel_values"].squeeze(0) 
        
        boxes = []
        labels = []
        masks = []
        
        if sample['annotations']:
            for ann in sample['annotations']:
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(int(ann.get('category_id', sample['label'])))
                    
                    # Gerar a máscara (LUNA16)
                    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
                    if 'segmentation' in ann and ann['segmentation']:
                        from PIL import ImageDraw
                        m_img = Image.new('L', (img_np.shape[1], img_np.shape[0]), 0)
                        ImageDraw.Draw(m_img).polygon(ann['segmentation'][0], outline=1, fill=1)
                        mask = np.array(m_img)
                    else:
                        # Fallback na ausência de máscara para simulação via bounding box
                        mask[int(y):int(y+h), int(x):int(x+w)] = 1
                    masks.append(mask)
        
        # Se não houver nódulos, inserimos arrays vazios obrigatórios
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, img_np.shape[0], img_np.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }
        
        return image_tensor, target
    
    def __len__(self):
        return len(self.samples)
    
    def get_sample_info(self, idx):
        """Retorna informações detalhadas sobre uma amostra."""
        if idx < len(self.samples):
            sample = self.samples[idx]
            return {
                "index": idx,
                "image_path": sample['image_path'],
                "filename": os.path.basename(sample['image_path']),
                "label": sample['label'],
                "category_name": sample['category_name'],
                "annotation_format": self.annotation_format,
                "annotations": sample.get('annotations', []),
                "image_info": sample.get('image_info', {})
            }
        return None
    
    def get_class_distribution(self):
        """Retorna a distribuição de classes no dataset."""
        if not self.samples:
            return {}
        
        labels = [sample['label'] for sample in self.samples]
        unique_labels = set(labels)
        distribution = {label: labels.count(label) for label in unique_labels}
        
        # Adicionar nomes das categorias se disponível
        labeled_distribution = {}
        for label, count in distribution.items():
            # Encontrar nome da categoria para este label
            category_name = None
            for sample in self.samples:
                if sample['label'] == label:
                    category_name = sample['category_name']
                    break
            
            labeled_distribution[f"{label} ({category_name})"] = count
        
        return labeled_distribution
    
    def visualize_sample(self, idx, show_annotations=True):
        """Visualiza amostra com as respectivas anotações (se COCO)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        sample_info = self.get_sample_info(idx)
        if not sample_info:
            print(f"Índice {idx} inválido")
            return
        
        # Carregar imagem original
        img = Image.open(sample_info['image_path']).convert("RGB")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        
        # Mostrar anotações COCO se disponível
        if show_annotations and sample_info['annotations']:
            for ann in sample_info['annotations']:
                if 'bbox' in ann:
                    # COCO bbox: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                
                if 'segmentation' in ann and ann['segmentation']:
                    # Mostrar segmentação se disponível
                    # Isso seria mais complexo, por agora apenas bbox
                    pass
        
        ax.set_title(f"Arquivo: {sample_info['filename']}\n"
                    f"Classe: {sample_info['label']} ({sample_info['category_name']})\n"
                    f"Formato: {sample_info['annotation_format']}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Mostrar informações detalhadas
        print(f"Informações da amostra {idx}:")
        print(f"   Arquivo: {sample_info['filename']}")
        print(f"   Classe: {sample_info['label']} ({sample_info['category_name']})")
        print(f"   Formato: {sample_info['annotation_format']}")
        if sample_info['annotations']:
            print(f"   Anotações: {len(sample_info['annotations'])} encontradas")
            for i, ann in enumerate(sample_info['annotations'][:3]):  # Mostrar até 3
                if 'bbox' in ann:
                    print(f"     {i+1}. BBox: {ann['bbox']}")
                if 'area' in ann:
                    print(f"        Área: {ann['area']}")


# Função utilitária para criar datasets facilmente
def create_luna_dataset(folder, image_processor, annotation_format='auto', coco_file=None, transform=None):
    """
    Cria um dataset LUNA com detecção automática do formato de anotação.
    
    Args:
        folder: Pasta com as imagens
        image_processor: Processador de imagens
        annotation_format: 'txt', 'coco', ou 'auto'
        coco_file: Caminho específico para arquivo COCO (opcional)
        transform: Transformações opcionais
    
    Returns:
        LunaDataset configurado
    """
    return LunaDataset(
        folder=folder,
        image_processor=image_processor,
        transform=transform,
        annotation_format=annotation_format,
        coco_file=coco_file
    )