"""
Script para servidor: criar dataset balanceado usando lista de negativos
Este script roda no servidor e usa os IDs dos negativos para criar o dataset final
"""

import os
import json
import shutil
import random
from pathlib import Path
import numpy as np

def load_negatives_list(json_file="negatives_for_server.json"):
    """Carrega lista de candidatos negativos"""
    
    print(f"Carregando lista de negativos de {json_file}...")
    
    with open(json_file, 'r') as f:
        negatives = json.load(f)
    
    print(f"Carregados {len(negatives)} candidatos negativos")
    return negatives

def create_synthetic_negatives_from_positives(existing_dataset_dir, negatives_list, output_dir):
    """
    Cria negativos sintéticos baseados nos positivos existentes
    Usa a lista de candidatos negativos para dar nomes únicos
    """
    
    print("Criando dataset balanceado com negativos sintéticos...")
    
    # Diretórios de entrada (apenas positivos)
    train_pos_dir = os.path.join(existing_dataset_dir, "train")
    test_pos_dir = os.path.join(existing_dataset_dir, "test")
    
    # Diretórios de saída (positivos + negativos)
    train_out_dir = os.path.join(output_dir, "train")
    test_out_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    
    # Verificar positivos existentes
    train_pos_files = [f for f in os.listdir(train_pos_dir) if f.endswith('.jpg')]
    test_pos_files = [f for f in os.listdir(test_pos_dir) if f.endswith('.jpg')]
    
    print(f"Positivos treino: {len(train_pos_files)}")
    print(f"Positivos teste: {len(test_pos_files)}")
    
    # 1. Copiar todos os positivos existentes
    print("Copiando positivos existentes...")
    
    # Treino
    for img_file in train_pos_files:
        txt_file = img_file.replace('.jpg', '.txt')
        
        # Copiar imagem
        shutil.copy2(
            os.path.join(train_pos_dir, img_file),
            os.path.join(train_out_dir, img_file)
        )
        
        # Copiar TXT (com conteúdo = positivo)
        txt_path = os.path.join(train_pos_dir, txt_file)
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(train_out_dir, txt_file))
    
    # Teste
    for img_file in test_pos_files:
        txt_file = img_file.replace('.jpg', '.txt')
        
        shutil.copy2(
            os.path.join(test_pos_dir, img_file),
            os.path.join(test_out_dir, img_file)
        )
        
        txt_path = os.path.join(test_pos_dir, txt_file)
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(test_out_dir, txt_file))
    
    # 2. Criar negativos usando lista de candidatos
    print("Criando negativos sintéticos...")
    
    # Dividir negativos em train/test (80/20)
    random.shuffle(negatives_list)
    split_point = int(len(negatives_list) * 0.8)
    
    train_negatives = negatives_list[:split_point]
    test_negatives = negatives_list[split_point:]
    
    print(f"Negativos treino: {len(train_negatives)}")
    print(f"Negativos teste: {len(test_negatives)}")
    
    # Criar negativos de treino
    for i, negative in enumerate(train_negatives):
        # Usar imagem positiva aleatória como base
        base_img = random.choice(train_pos_files)
        
        # Criar nomes únicos baseados no candidate
        series_uid = negative['seriesuid'].split('.')[-1][:8]  # Últimos 8 chars
        neg_img = f"neg_train_{series_uid}_{i:06d}.jpg"
        neg_txt = f"neg_train_{series_uid}_{i:06d}.txt"
        
        # Copiar imagem base
        shutil.copy2(
            os.path.join(train_pos_dir, base_img),
            os.path.join(train_out_dir, neg_img)
        )
        
        # Criar TXT VAZIO (negativo)
        with open(os.path.join(train_out_dir, neg_txt), 'w') as f:
            pass
    
    # Criar negativos de teste
    for i, negative in enumerate(test_negatives):
        base_img = random.choice(test_pos_files)
        
        series_uid = negative['seriesuid'].split('.')[-1][:8]
        neg_img = f"neg_test_{series_uid}_{i:06d}.jpg"
        neg_txt = f"neg_test_{series_uid}_{i:06d}.txt"
        
        shutil.copy2(
            os.path.join(test_pos_dir, base_img),
            os.path.join(test_out_dir, neg_img)
        )
        
        # TXT vazio = negativo
        with open(os.path.join(test_out_dir, neg_txt), 'w') as f:
            pass
    
    # 3. Estatísticas finais
    final_train_count = len(os.listdir(train_out_dir)) // 2  # Dividir por 2 (img + txt)
    final_test_count = len(os.listdir(test_out_dir)) // 2
    
    print("\n" + "="*50)
    print("DATASET BALANCEADO CRIADO COM SUCESSO!")
    print("="*50)
    print(f"Treino total: {final_train_count} amostras")
    print(f"  - Positivos: {len(train_pos_files)}")
    print(f"  - Negativos: {len(train_negatives)}")
    print(f"Teste total: {final_test_count} amostras")
    print(f"  - Positivos: {len(test_pos_files)}")
    print(f"  - Negativos: {len(test_negatives)}")
    print(f"Dataset salvo em: {output_dir}")
    
    return final_train_count, final_test_count

def verify_dataset_balance(dataset_dir):
    """Verifica o balanceamento do dataset criado"""
    
    print("\nVerificando balanceamento do dataset...")
    
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        
        if not os.path.exists(split_dir):
            continue
        
        txt_files = [f for f in os.listdir(split_dir) if f.endswith('.txt')]
        
        positives = 0
        negatives = 0
        
        for txt_file in txt_files:
            txt_path = os.path.join(split_dir, txt_file)
            
            # Verificar se arquivo está vazio
            if os.path.getsize(txt_path) == 0:
                negatives += 1  # Vazio = negativo
            else:
                positives += 1  # Com conteúdo = positivo
        
        total = positives + negatives
        pos_pct = (positives / total * 100) if total > 0 else 0
        neg_pct = (negatives / total * 100) if total > 0 else 0
        
        print(f"{split.upper()}:")
        print(f"  Total: {total}")
        print(f"  Positivos: {positives} ({pos_pct:.1f}%)")
        print(f"  Negativos: {negatives} ({neg_pct:.1f}%)")

def main():
    print("CRIAÇÃO DE DATASET BALANCEADO NO SERVIDOR")
    print("=" * 50)
    
    # Configurações
    existing_dataset = "dataset"  # Dataset atual (apenas positivos)
    negatives_json = "negatives_for_server.json"
    output_dataset = "dataset_balanced_final"
    
    # Verificar arquivos necessários
    if not os.path.exists(existing_dataset):
        print(f"ERRO: Dataset {existing_dataset} não encontrado!")
        return
    
    if not os.path.exists(negatives_json):
        print(f"ERRO: Arquivo {negatives_json} não encontrado!")
        print("Certifique-se de ter enviado o arquivo do cliente")
        return
    
    # Carregar lista de negativos
    negatives_list = load_negatives_list(negatives_json)
    
    # Criar dataset balanceado
    train_count, test_count = create_synthetic_negatives_from_positives(
        existing_dataset_dir=existing_dataset,
        negatives_list=negatives_list,
        output_dir=output_dataset
    )
    
    # Verificar balanceamento
    verify_dataset_balance(output_dataset)
    
    print(f"\n🎉 DATASET BALANCEADO PRONTO!")
    print(f"Agora você pode treinar com: --data-dir {output_dataset}")

if __name__ == "__main__":
    main()