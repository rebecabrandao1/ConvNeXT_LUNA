"""
Script simplificado para adicionar dados negativos ao dataset existente
Cria arquivos TXT vazios para simular negativos
"""

import os
import random
import shutil
from pathlib import Path
import pandas as pd

def create_synthetic_negatives(existing_dataset_dir, output_dir, negative_ratio=1.0):
    """
    Cria negativos sintéticos baseados nos positivos existentes
    negative_ratio: proporção de negativos para positivos (1.0 = mesmo número)
    """
    
    print("Criando dataset com negativos sintéticos...")
    
    # Diretórios de entrada
    train_pos_dir = os.path.join(existing_dataset_dir, "train")
    test_pos_dir = os.path.join(existing_dataset_dir, "test") 
    
    # Diretórios de saída
    train_out_dir = os.path.join(output_dir, "train")
    test_out_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    
    # Processar train
    train_files = [f for f in os.listdir(train_pos_dir) if f.endswith('.jpg')]
    print(f"Positivos de treino encontrados: {len(train_files)}")
    
    # Copiar positivos existentes
    for img_file in train_files:
        txt_file = img_file.replace('.jpg', '.txt')
        
        # Copiar imagem
        shutil.copy2(
            os.path.join(train_pos_dir, img_file),
            os.path.join(train_out_dir, img_file)
        )
        
        # Copiar TXT (que já tem conteúdo = positivo)
        if os.path.exists(os.path.join(train_pos_dir, txt_file)):
            shutil.copy2(
                os.path.join(train_pos_dir, txt_file),
                os.path.join(train_out_dir, txt_file)
            )
    
    # Criar negativos baseados nos positivos
    num_negatives = int(len(train_files) * negative_ratio)
    print(f"Criando {num_negatives} negativos de treino...")
    
    for i in range(num_negatives):
        # Selecionar uma imagem positiva aleatória para usar como base
        base_img = random.choice(train_files)
        base_txt = base_img.replace('.jpg', '.txt')
        
        # Nomes para o negativo
        neg_img = f"negative_train_{i:06d}.jpg"
        neg_txt = f"negative_train_{i:06d}.txt"
        
        # Copiar imagem (mesma imagem, mas com label diferente)
        shutil.copy2(
            os.path.join(train_pos_dir, base_img),
            os.path.join(train_out_dir, neg_img)
        )
        
        # Criar TXT VAZIO para indicar negativo
        with open(os.path.join(train_out_dir, neg_txt), 'w') as f:
            pass  # Arquivo vazio = classe 0 (negativo)
    
    # Processar test (mesmo processo)
    test_files = [f for f in os.listdir(test_pos_dir) if f.endswith('.jpg')]
    print(f"Positivos de teste encontrados: {len(test_files)}")
    
    # Copiar positivos de teste
    for img_file in test_files:
        txt_file = img_file.replace('.jpg', '.txt')
        
        shutil.copy2(
            os.path.join(test_pos_dir, img_file),
            os.path.join(test_out_dir, img_file)
        )
        
        if os.path.exists(os.path.join(test_pos_dir, txt_file)):
            shutil.copy2(
                os.path.join(test_pos_dir, txt_file),
                os.path.join(test_out_dir, txt_file)
            )
    
    # Criar negativos de teste
    num_test_negatives = int(len(test_files) * negative_ratio)
    print(f"Criando {num_test_negatives} negativos de teste...")
    
    for i in range(num_test_negatives):
        base_img = random.choice(test_files)
        
        neg_img = f"negative_test_{i:06d}.jpg"
        neg_txt = f"negative_test_{i:06d}.txt"
        
        shutil.copy2(
            os.path.join(test_pos_dir, base_img),
            os.path.join(test_out_dir, neg_img)
        )
        
        # TXT vazio = negativo
        with open(os.path.join(test_out_dir, neg_txt), 'w') as f:
            pass
    
    print("Dataset com negativos criado com sucesso!")
    print(f"Train: {len(train_files)} positivos + {num_negatives} negativos = {len(train_files) + num_negatives}")
    print(f"Test: {len(test_files)} positivos + {num_test_negatives} negativos = {len(test_files) + num_test_negatives}")

def analyze_candidates_csv(candidates_path):
    """Analisa o arquivo candidates.csv para entender sua estrutura"""
    
    print("Analisando candidates.csv...")
    
    try:
        # Ler apenas primeiras linhas
        df_sample = pd.read_csv(candidates_path, nrows=10)
        print("Estrutura do arquivo:")
        print(f"Colunas: {df_sample.columns.tolist()}")
        print("Primeiras linhas:")
        print(df_sample)
        
        # Estatísticas gerais
        df_full = pd.read_csv(candidates_path)
        print(f"\nTotal de linhas: {len(df_full)}")
        
        if 'class' in df_full.columns:
            class_counts = df_full['class'].value_counts()
            print("Distribuição de classes:")
            print(class_counts)
        
        return df_full
        
    except Exception as e:
        print(f"Erro ao analisar candidates.csv: {e}")
        return None

def main():
    print("CRIAÇÃO DE DATASET BALANCEADO COM NEGATIVOS")
    print("=" * 50)
    
    # Configurações
    existing_dataset = "dataset_balanced"  # Dataset atual apenas com positivos
    output_dataset = "dataset_balanced_with_negatives"  # Novo dataset balanceado
    candidates_path = "dataset/candidates.csv"
    
    # Verificar se dataset atual existe
    if not os.path.exists(existing_dataset):
        print(f"ERRO: Dataset {existing_dataset} não encontrado!")
        return
    
    # Analisar candidates.csv se existir
    if os.path.exists(candidates_path):
        candidates_df = analyze_candidates_csv(candidates_path)
        
        if candidates_df is not None and 'class' in candidates_df.columns:
            print("\nOpção disponível: usar dados reais de candidates.csv")
            print("Porém, para simplicidade, vamos criar negativos sintéticos primeiro")
    
    # Criar dataset com negativos sintéticos
    create_synthetic_negatives(
        existing_dataset_dir=existing_dataset,
        output_dir=output_dataset,
        negative_ratio=1.0  # Mesmo número de negativos que positivos
    )
    
    print(f"\nDataset balanceado criado em: {output_dataset}")
    print("Agora você pode treinar com dados balanceados!")

if __name__ == "__main__":
    main()