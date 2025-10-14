"""
Script para processar dados negativos do LUNA16 a partir dos subsets e candidates.csv
"""

import os
import pandas as pd
import numpy as np
import zipfile
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import cv2

def extract_subsets(dataset_dir, extract_dir):
    """Extrai todos os subsets zipados"""
    print("Extraindo subsets...")
    
    for i in range(10):  # subsets 0-9
        subset_zip = os.path.join(dataset_dir, f"subset{i}.zip")
        if os.path.exists(subset_zip):
            print(f"Extraindo subset{i}.zip...")
            with zipfile.ZipFile(subset_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"Aviso: subset{i}.zip não encontrado")

def load_candidates(candidates_path):
    """Carrega o arquivo candidates.csv"""
    print(f"Carregando candidatos de {candidates_path}...")
    
    # Ler apenas as primeiras linhas para entender a estrutura
    df = pd.read_csv(candidates_path, nrows=5)
    print("Colunas encontradas:", df.columns.tolist())
    print("Primeiras linhas:")
    print(df.head())
    
    # Carregar dados completos
    df_full = pd.read_csv(candidates_path)
    print(f"Total de candidatos: {len(df_full)}")
    
    return df_full

def load_annotations(annotations_path):
    """Carrega as anotações (nódulos verdadeiros)"""
    print(f"Carregando anotações de {annotations_path}...")
    
    df = pd.read_csv(annotations_path)
    print("Colunas das anotações:", df.columns.tolist())
    print(f"Total de nódulos anotados: {len(df)}")
    
    return df

def create_balanced_dataset(candidates_df, annotations_df, subsets_dir, output_dir, max_negatives=3255):
    """Cria dataset balanceado com positivos e negativos"""
    
    print("Criando dataset balanceado...")
    
    # Criar diretórios de saída
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Identificar candidatos negativos
    # Assumindo que candidates.csv tem coluna 'class' onde 0=negativo, 1=positivo
    if 'class' in candidates_df.columns:
        negatives = candidates_df[candidates_df['class'] == 0]
        positives = candidates_df[candidates_df['class'] == 1]
    else:
        # Se não tem coluna class, usar coordenadas para comparar com annotations
        print("Identificando negativos por comparação com anotações...")
        negatives = identify_negatives_by_coordinates(candidates_df, annotations_df)
        positives = candidates_df[~candidates_df.index.isin(negatives.index)]
    
    print(f"Candidatos positivos: {len(positives)}")
    print(f"Candidatos negativos: {len(negatives)}")
    
    # Limitar número de negativos para balancear
    if len(negatives) > max_negatives:
        negatives = negatives.sample(n=max_negatives, random_state=42)
        print(f"Limitando negativos para {max_negatives} amostras")
    
    # Processar negativos
    negative_count = 0
    for idx, row in tqdm(negatives.iterrows(), desc="Processando negativos", total=len(negatives)):
        
        # Encontrar arquivo de imagem correspondente
        series_uid = row['seriesuid']
        
        # Procurar arquivo .mhd nos subsets extraídos
        mhd_file = find_mhd_file(subsets_dir, series_uid)
        
        if mhd_file:
            # Extrair patch da região do candidato
            patch = extract_patch_from_mhd(mhd_file, row)
            
            if patch is not None:
                # Salvar imagem e label
                img_name = f"negative_{negative_count:06d}.jpg"
                txt_name = f"negative_{negative_count:06d}.txt"
                
                # Dividir em train/test (80/20)
                if negative_count % 5 == 0:  # 20% para test
                    img_path = os.path.join(test_dir, img_name)
                    txt_path = os.path.join(test_dir, txt_name)
                else:  # 80% para train
                    img_path = os.path.join(train_dir, img_name)
                    txt_path = os.path.join(train_dir, txt_name)
                
                # Salvar imagem
                cv2.imwrite(img_path, patch)
                
                # Criar arquivo TXT vazio (label 0 = negativo)
                with open(txt_path, 'w') as f:
                    pass  # Arquivo vazio indica negativo
                
                negative_count += 1
        
        # Limitar para não processar demais
        if negative_count >= max_negatives:
            break
    
    print(f"Processados {negative_count} negativos")
    return negative_count

def identify_negatives_by_coordinates(candidates_df, annotations_df, threshold=5.0):
    """Identifica negativos comparando coordenadas com anotações"""
    
    negatives = []
    
    for idx, candidate in tqdm(candidates_df.iterrows(), desc="Identificando negativos"):
        series_uid = candidate['seriesuid']
        coord_x = candidate['coordX']
        coord_y = candidate['coordY'] 
        coord_z = candidate['coordZ']
        
        # Verificar se há anotação próxima
        series_annotations = annotations_df[annotations_df['seriesuid'] == series_uid]
        
        is_negative = True
        for _, annotation in series_annotations.iterrows():
            # Calcular distância euclidiana
            dist = np.sqrt(
                (coord_x - annotation['coordX'])**2 + 
                (coord_y - annotation['coordY'])**2 + 
                (coord_z - annotation['coordZ'])**2
            )
            
            if dist <= threshold:  # Dentro do threshold = positivo
                is_negative = False
                break
        
        if is_negative:
            negatives.append(idx)
    
    return candidates_df.loc[negatives]

def find_mhd_file(subsets_dir, series_uid):
    """Encontra arquivo .mhd correspondente ao series_uid"""
    
    for root, dirs, files in os.walk(subsets_dir):
        for file in files:
            if file.endswith('.mhd') and series_uid in file:
                return os.path.join(root, file)
    
    return None

def extract_patch_from_mhd(mhd_file, candidate_row, patch_size=64):
    """Extrai patch 2D de arquivo MHD nas coordenadas do candidato"""
    
    try:
        import SimpleITK as sitk
        
        # Ler imagem
        image = sitk.ReadImage(mhd_file)
        image_array = sitk.GetArrayFromImage(image)
        
        # Converter coordenadas mundo para pixel
        coord_x = candidate_row['coordX']
        coord_y = candidate_row['coordY'] 
        coord_z = candidate_row['coordZ']
        
        # Transformar coordenadas (simplificado - pode precisar ajustar)
        pixel_coords = image.TransformPhysicalPointToIndex([coord_x, coord_y, coord_z])
        
        z, y, x = pixel_coords
        
        # Extrair patch 2D
        if (z >= 0 and z < image_array.shape[0] and 
            y >= patch_size//2 and y < image_array.shape[1] - patch_size//2 and
            x >= patch_size//2 and x < image_array.shape[2] - patch_size//2):
            
            patch = image_array[z, 
                              y - patch_size//2:y + patch_size//2,
                              x - patch_size//2:x + patch_size//2]
            
            # Normalizar e converter para uint8
            patch = ((patch - patch.min()) / (patch.max() - patch.min()) * 255).astype(np.uint8)
            
            # Redimensionar para 224x224 (tamanho esperado pelo modelo)
            patch = cv2.resize(patch, (224, 224))
            
            return patch
    
    except Exception as e:
        print(f"Erro ao processar {mhd_file}: {e}")
        return None

def main():
    # Configurações
    dataset_dir = "dataset"
    subsets_extract_dir = "subsets_extracted"
    output_dir = "dataset_with_negatives"
    
    candidates_path = os.path.join(dataset_dir, "candidates.csv")
    annotations_path = os.path.join(dataset_dir, "annotations (1).csv")
    
    print("PROCESSAMENTO DE DADOS NEGATIVOS LUNA16")
    print("=" * 50)
    
    # 1. Extrair subsets
    if not os.path.exists(subsets_extract_dir):
        extract_subsets(dataset_dir, subsets_extract_dir)
    else:
        print("Subsets já extraídos")
    
    # 2. Carregar candidatos e anotações
    try:
        candidates_df = load_candidates(candidates_path)
        annotations_df = load_annotations(annotations_path)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    # 3. Criar dataset balanceado
    try:
        negative_count = create_balanced_dataset(
            candidates_df, annotations_df, 
            subsets_extract_dir, output_dir
        )
        
        print(f"\nDataset criado com sucesso!")
        print(f"Negativos processados: {negative_count}")
        print(f"Dataset salvo em: {output_dir}")
        
    except Exception as e:
        print(f"Erro ao criar dataset: {e}")

if __name__ == "__main__":
    main()