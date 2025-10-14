"""
Processador LUNA16 para servidor
"""

import os
import zipfile
import tempfile
import shutil
import numpy as np
from PIL import Image
import SimpleITK as sitk
import argparse
from pathlib import Path
import json

class Luna16Processor:
    def __init__(self, output_dir="dataset_balanced", balance_ratio=1.0):
        """
        Inicializa o processador LUNA16.
        
        Args:
            output_dir: Diretorio de saida
            balance_ratio: Proporcao negativo/positivo (1.0 = balanceado)
        """
        self.output_dir = output_dir
        self.balance_ratio = balance_ratio
        
        # Criar estrutura de diretorios
        self.train_dir = os.path.join(output_dir, "train")
        self.test_dir = os.path.join(output_dir, "test")
        
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        print(f"Processador LUNA16 inicializado")
        print(f"   Saida: {output_dir}")
        print(f"   Balance ratio: {balance_ratio}")
    
    def process_all_subsets(self, subset_dir, max_subsets=None):
        """
        Processa todos os subsets encontrados no diretorio.
        
        Args:
            subset_dir: Diretorio contendo os ZIPs dos subsets
            max_subsets: Numero maximo de subsets a processar
        """
        print(f"Processando subsets em: {subset_dir}")
        
        # Encontrar arquivos ZIP
        zip_files = []
        for file in os.listdir(subset_dir):
            if file.startswith("subset") and file.endswith(".zip"):
                zip_files.append(file)
        
        zip_files.sort()  # subset0, subset1, etc.
        
        if max_subsets:
            zip_files = zip_files[:max_subsets]
        
        print(f"Encontrados {len(zip_files)} subsets: {zip_files}")
        
        total_samples = 0
        subset_stats = {}
        
        for i, zip_file in enumerate(zip_files):
            print(f"\n[{i+1}/{len(zip_files)}] Processando {zip_file}...")
            
            zip_path = os.path.join(subset_dir, zip_file)
            subset_name = os.path.splitext(zip_file)[0]
            
            try:
                samples = self.process_subset(zip_path, subset_name)
                total_samples += samples
                subset_stats[subset_name] = samples
                print(f"   Geradas {samples} amostras do {subset_name}")
                
            except Exception as e:
                print(f"   ERRO ao processar {zip_file}: {e}")
                continue
        
        print(f"\nProcessamento concluido:")
        print(f"   Total de amostras: {total_samples}")
        print(f"   Subsets processados: {len(subset_stats)}")
        
        # Salvar estatisticas
        stats_file = os.path.join(self.output_dir, "processing_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'total_samples': total_samples,
                'subset_stats': subset_stats,
                'balance_ratio': self.balance_ratio
            }, f, indent=2)
        
        print(f"   Estatisticas salvas em: {stats_file}")
        
        return total_samples
    
    def process_subset(self, zip_path, subset_name):
        """
        Processa um subset especifico.
        
        Args:
            zip_path: Caminho para o arquivo ZIP
            subset_name: Nome do subset
        
        Returns:
            int: Numero de amostras geradas
        """
        total_samples = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extrair ZIP
            print(f"   Extraindo {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Encontrar pasta do subset
            subset_path = None
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path) and subset_name in item.lower():
                    subset_path = item_path
                    break
            
            if not subset_path:
                subset_path = temp_dir
            
            # Processar volumes
            volume_files = self._find_volume_files(subset_path)
            print(f"   Encontrados {len(volume_files)} volumes")
            
            for volume_file in volume_files:
                try:
                    samples = self.process_volume(volume_file, subset_name)
                    total_samples += samples
                except Exception as e:
                    print(f"   AVISO: Erro ao processar {os.path.basename(volume_file)}: {e}")
                    continue
        
        return total_samples
    
    def _find_volume_files(self, directory):
        """Encontra arquivos de volume (.mhd) no diretorio."""
        volume_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.mhd'):
                    volume_files.append(os.path.join(root, file))
        
        return volume_files
    
    def process_volume(self, volume_path, subset_name):
        """
        Processa um volume individual.
        
        Args:
            volume_path: Caminho para arquivo .mhd
            subset_name: Nome do subset
            
        Returns:
            int: Numero de slices extraidos
        """
        # Carregar volume
        image = sitk.ReadImage(volume_path)
        volume_array = sitk.GetArrayFromImage(image)
        
        # Normalizar
        volume_array = self._normalize_volume(volume_array)
        
        # Gerar nome base
        volume_name = os.path.splitext(os.path.basename(volume_path))[0]
        
        # Extrair slices (todas sao negativas, pois nao temos anotacoes)
        num_slices = volume_array.shape[0]
        samples_to_extract = min(15, num_slices)  # Maximo 15 slices por volume
        
        extracted = 0
        step = max(1, num_slices // samples_to_extract)
        
        for i in range(0, num_slices, step):
            if extracted >= samples_to_extract:
                break
                
            slice_array = volume_array[i]
            
            # Verificar se slice tem conteudo relevante
            if not self._is_valid_slice(slice_array):
                continue
            
            # Converter para imagem
            slice_pil = self._array_to_pil(slice_array)
            
            # Nome do arquivo
            filename = f"img_{volume_name}_{i:03d}"
            
            # Determinar se vai para train ou test (80/20)
            is_test = extracted % 5 == 0
            target_dir = self.test_dir if is_test else self.train_dir
            
            # Salvar imagem
            img_path = os.path.join(target_dir, f"{filename}.jpg")
            slice_pil.save(img_path, quality=95)
            
            # Criar arquivo de anotacao vazio (sem nodulo)
            txt_path = os.path.join(target_dir, f"{filename}.txt")
            with open(txt_path, 'w') as f:
                f.write("")  # Arquivo vazio = sem nodulo
            
            extracted += 1
        
        return extracted
    
    def _normalize_volume(self, volume_array):
        """Normaliza o volume para 0-255."""
        # Clipar valores extremos
        p1, p99 = np.percentile(volume_array, [1, 99])
        volume_array = np.clip(volume_array, p1, p99)
        
        # Normalizar para 0-255
        volume_min = volume_array.min()
        volume_max = volume_array.max()
        
        if volume_max > volume_min:
            volume_array = ((volume_array - volume_min) / (volume_max - volume_min) * 255)
        
        return volume_array.astype(np.uint8)
    
    def _is_valid_slice(self, slice_array):
        """Verifica se o slice tem conteudo relevante."""
        # Verificar se nao e muito escuro ou muito claro
        mean_intensity = np.mean(slice_array)
        std_intensity = np.std(slice_array)
        
        return 20 < mean_intensity < 200 and std_intensity > 5
    
    def _array_to_pil(self, array):
        """Converte array numpy para imagem PIL."""
        # Garantir que esta no range 0-255
        array = np.clip(array, 0, 255).astype(np.uint8)
        
        # Converter para PIL
        pil_image = Image.fromarray(array, mode='L')
        
        # Redimensionar para 224x224
        pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return pil_image

def main():
    """Funcao principal."""
    parser = argparse.ArgumentParser(description='Processar subsets LUNA16 para o servidor')
    parser.add_argument('--subset-dir', default='subsets', 
                       help='Diretorio contendo os ZIPs dos subsets')
    parser.add_argument('--output-dir', default='dataset_balanced',
                       help='Diretorio de saida')
    parser.add_argument('--max-subsets', type=int, default=None,
                       help='Numero maximo de subsets a processar')
    parser.add_argument('--balance-ratio', type=float, default=1.0,
                       help='Proporcao negativo/positivo')
    
    args = parser.parse_args()
    
    print("PROCESSADOR LUNA16 - SERVIDOR")
    print("=" * 40)
    print(f"Diretorio subsets: {args.subset_dir}")
    print(f"Diretorio saida: {args.output_dir}")
    print(f"Max subsets: {args.max_subsets or 'Todos'}")
    print("=" * 40)
    
    # Verificar diretorio de entrada
    if not os.path.exists(args.subset_dir):
        print(f"ERRO: Diretorio nao encontrado: {args.subset_dir}")
        print("Coloque os arquivos subset*.zip no diretorio especificado")
        return
    
    # Processar
    processor = Luna16Processor(
        output_dir=args.output_dir,
        balance_ratio=args.balance_ratio
    )
    
    total_samples = processor.process_all_subsets(
        args.subset_dir,
        max_subsets=args.max_subsets
    )
    
    print(f"\nProcessamento concluido com sucesso!")
    print(f"Total de {total_samples} amostras geradas")
    print(f"Dados salvos em: {args.output_dir}")
    print(f"\nProximo passo:")
    print(f"  python train_balanced_servidor.py --epochs 400 --patience 50")

if __name__ == "__main__":
    main()