"""
Dataset balanceado para LUNA16 com amostras positivas e negativas
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path

class BalancedLunaDataset(Dataset):
    """
    Dataset balanceado com amostras positivas (com nódulos) e negativas (sem nódulos).
    """
    
    def __init__(self, data_dir, image_processor, split='train', transform=None):
        """
        Args:
            data_dir: Pasta com dados balanceados (ex: "dataset_balanced")
            image_processor: Processador de imagens do modelo
            split: 'train' ou 'test'
            transform: Transformações opcionais
        """
        self.image_processor = image_processor
        self.transform = transform
        self.samples = []
        
        # Pasta específica do split
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists():
            raise ValueError(f"Diretório não encontrado: {split_dir}")
        
        print(f"📊 Carregando dados balanceados - Split: {split}")
        print(f"   Diretório: {split_dir}")
        
        # Carregar todas as imagens e labels
        self._load_balanced_data(split_dir)
        
        print(f"✅ Total de amostras carregadas: {len(self.samples)}")
        self._print_statistics()
    
    def _load_balanced_data(self, split_dir):
        """Carrega dados da pasta balanceada."""
        
        # Processar todas as imagens .jpg
        for img_file in split_dir.glob("*.jpg"):
            txt_file = img_file.with_suffix('.txt')
            
            if not txt_file.exists():
                print(f"⚠️ Arquivo TXT não encontrado para {img_file.name}")
                continue
            
            # Determinar label baseado no conteúdo do arquivo TXT
            try:
                with open(txt_file, 'r') as f:
                    txt_content = f.read().strip()
                
                if txt_content:
                    # Arquivo TXT com conteúdo = amostra positiva (com nódulo)
                    label = 1
                    # Parsear bboxes se necessário
                    bboxes = self._parse_yolo_annotations(txt_content)
                else:
                    # Arquivo TXT vazio = amostra negativa (sem nódulo)
                    label = 0
                    bboxes = []
                
                # Extrair informações do nome do arquivo
                filename = img_file.name
                
                # Tentar extrair series_uid e slice_num do nome
                series_uid = ""
                slice_num = -1
                
                if filename.startswith('img_'):
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 3:
                        series_uid = '_'.join(parts[1:-1])
                        try:
                            slice_num = int(parts[-1])
                        except ValueError:
                            pass
                
                # Adicionar amostra
                sample_info = {
                    'image_path': str(img_file),
                    'label': label,
                    'filename': filename,
                    'series_uid': series_uid,
                    'slice_num': slice_num,
                    'bboxes': bboxes,
                    'source': 'balanced_dataset'
                }
                
                self.samples.append(sample_info)
                
            except Exception as e:
                print(f"❌ Erro ao processar {img_file.name}: {e}")
        
        print(f"   ✅ {len(self.samples)} amostras processadas")
    
    def _parse_yolo_annotations(self, txt_content):
        """
        Parsear anotações no formato YOLO.
        
        Args:
            txt_content: Conteúdo do arquivo TXT
            
        Returns:
            Lista de bounding boxes
        """
        
        bboxes = []
        
        for line in txt_content.strip().split('\n'):
            if line.strip():
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        bbox = {
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        }
                        bboxes.append(bbox)
                        
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Erro ao parsear linha: {line} - {e}")
        
        return bboxes
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Carregar imagem
            img = Image.open(sample['image_path']).convert("L")  # Escala de cinza
            img_np = np.expand_dims(np.array(img), axis=-1)  # (H, W, 1)
            
            # Aplicar transformações se fornecidas
            if self.transform:
                img_np = self.transform(img_np)
            
            # Processar com o image_processor
            processed = self.image_processor(
                [img_np], return_tensors="pt", image_mean=[0.5], image_std=[0.5]
            )
            
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "labels": torch.tensor(sample['label'], dtype=torch.long),
                "filename": sample['filename'],
                "series_uid": sample['series_uid'],
                "slice_num": sample['slice_num'],
                "source": sample['source']
            }
        
        except Exception as e:
            print(f"❌ Erro ao carregar {sample['image_path']}: {e}")
            # Retornar tensor vazio em caso de erro
            return {
                "pixel_values": torch.zeros((1, 224, 224)),
                "labels": torch.tensor(0, dtype=torch.long),
                "filename": sample['filename'],
                "series_uid": sample['series_uid'],
                "slice_num": sample['slice_num'],
                "source": "error"
            }
    
    def __len__(self):
        return len(self.samples)
    
    def _print_statistics(self):
        """Imprime estatísticas do dataset."""
        if not self.samples:
            return
        
        # Contar positivos e negativos
        positive_count = sum(1 for s in self.samples if s['label'] == 1)
        negative_count = sum(1 for s in self.samples if s['label'] == 0)
        total_count = len(self.samples)
        
        print(f"\n📊 Estatísticas do Dataset:")
        print(f"   Total: {total_count} amostras")
        print(f"   Com nódulo (positivos): {positive_count} ({positive_count/total_count*100:.1f}%)")
        print(f"   Sem nódulo (negativos): {negative_count} ({negative_count/total_count*100:.1f}%)")
        
        # Estatísticas por series_uid
        series_stats = defaultdict(lambda: {'total': 0, 'positive': 0, 'negative': 0})
        
        for sample in self.samples:
            series_uid = sample.get('series_uid', 'unknown')
            series_stats[series_uid]['total'] += 1
            
            if sample['label'] == 1:
                series_stats[series_uid]['positive'] += 1
            else:
                series_stats[series_uid]['negative'] += 1
        
        print(f"\n📋 Por volume (series_uid):")
        for series_uid, stats in list(series_stats.items())[:5]:  # Mostrar apenas os primeiros 5
            total = stats['total']
            pos = stats['positive']
            neg = stats['negative']
            print(f"   {series_uid[:50]}...: {total} amostras ({pos} pos, {neg} neg)")
        
        if len(series_stats) > 5:
            print(f"   ... e mais {len(series_stats) - 5} volumes")
    
    def get_class_distribution(self):
        """Retorna distribuição de classes."""
        distribution = defaultdict(int)
        
        for sample in self.samples:
            distribution[sample['label']] += 1
        
        return dict(distribution)
    
    def get_sample_info(self, idx):
        """Retorna informações detalhadas de uma amostra."""
        if 0 <= idx < len(self.samples):
            return self.samples[idx].copy()
        return None


def create_balanced_datasets(data_dir="dataset_balanced", image_processor=None):
    """
    Cria datasets balanceados de treino e teste.
    
    Args:
        data_dir: Pasta com dados balanceados
        image_processor: Processador de imagens
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    
    if image_processor is None:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
    
    print(f"🚀 Criando datasets balanceados...")
    
    # Dataset de treino
    train_dataset = BalancedLunaDataset(
        data_dir=data_dir,
        image_processor=image_processor,
        split='train'
    )
    
    # Dataset de teste
    test_dataset = BalancedLunaDataset(
        data_dir=data_dir,
        image_processor=image_processor,
        split='test'
    )
    
    return train_dataset, test_dataset


def demo_balanced_dataset():
    """Demonstração de uso do dataset balanceado."""
    
    from transformers import AutoImageProcessor
    
    # Criar processador
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
    
    # Criar datasets balanceados
    train_ds, test_ds = create_balanced_datasets(
        data_dir="dataset_balanced",
        image_processor=image_processor
    )
    
    print(f"\n🔍 Exemplo de uso:")
    print(f"   Dataset treino: {len(train_ds)} amostras")
    print(f"   Dataset teste: {len(test_ds)} amostras")
    
    # Mostrar distribuição de classes
    print(f"\n📊 Distribuição treino:")
    train_dist = train_ds.get_class_distribution()
    for label, count in train_dist.items():
        label_name = "Com nódulo" if label == 1 else "Sem nódulo"
        print(f"   {label_name} (classe {label}): {count} amostras")
    
    print(f"\n📊 Distribuição teste:")
    test_dist = test_ds.get_class_distribution()
    for label, count in test_dist.items():
        label_name = "Com nódulo" if label == 1 else "Sem nódulo"
        print(f"   {label_name} (classe {label}): {count} amostras")
    
    # Testar primeira amostra se disponível
    if len(train_ds) > 0:
        sample = train_ds[0]
        sample_info = train_ds.get_sample_info(0)
        
        print(f"\n📋 Primeira amostra:")
        print(f"   Arquivo: {sample_info['filename']}")
        print(f"   Label: {sample_info['label']} ({'Com nódulo' if sample_info['label'] == 1 else 'Sem nódulo'})")
        print(f"   Series UID: {sample_info['series_uid']}")
        print(f"   Slice: {sample_info['slice_num']}")
        print(f"   Shape: {sample['pixel_values'].shape}")
        print(f"   Bboxes: {len(sample_info['bboxes'])} encontrados")


if __name__ == "__main__":
    demo_balanced_dataset()