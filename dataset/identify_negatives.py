"""
Script para identificar candidatos negativos do candidates.csv
e gerar lista de negativos para processamento no servidor
"""

import pandas as pd
import json
import os
from pathlib import Path

def analyze_candidates_and_annotations():
    """Analisa candidates.csv e annotations.csv para identificar negativos"""
    
    print("Analisando candidates.csv e annotations.csv...")
    
    # Carregar dados
    candidates_path = "candidates.csv"
    annotations_path = "annotations (1).csv"
    
    if not os.path.exists(candidates_path):
        print(f"Erro: {candidates_path} não encontrado!")
        return None
        
    if not os.path.exists(annotations_path):
        print(f"Erro: {annotations_path} não encontrado!")
        return None
    
    # Ler candidates (primeiras linhas para ver estrutura)
    print("Estrutura do candidates.csv:")
    candidates_sample = pd.read_csv(candidates_path, nrows=5)
    print("Colunas:", candidates_sample.columns.tolist())
    print(candidates_sample)
    
    # Ler annotations
    print("\nEstrutura do annotations.csv:")
    annotations_df = pd.read_csv(annotations_path)
    print("Colunas:", annotations_df.columns.tolist())
    print(f"Total de anotações: {len(annotations_df)}")
    
    # Carregar candidates completo
    candidates_df = pd.read_csv(candidates_path)
    print(f"\nTotal de candidatos: {len(candidates_df)}")
    
    return candidates_df, annotations_df

def identify_negatives_by_class(candidates_df):
    """Se candidates.csv tem coluna 'class', usar diretamente"""
    
    if 'class' in candidates_df.columns:
        negatives = candidates_df[candidates_df['class'] == 0]
        positives = candidates_df[candidates_df['class'] == 1]
        
        print(f"Negativos (class=0): {len(negatives)}")
        print(f"Positivos (class=1): {len(positives)}")
        
        return negatives, positives
    else:
        print("Coluna 'class' não encontrada - será necessário usar coordenadas")
        return None, None

def identify_negatives_by_coordinates(candidates_df, annotations_df, threshold=5.0):
    """Identifica negativos comparando distância com anotações"""
    
    print(f"Identificando negativos por distância (threshold={threshold}mm)...")
    
    import numpy as np
    from tqdm import tqdm
    
    negative_indices = []
    
    for idx, candidate in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Processando candidatos"):
        series_uid = candidate['seriesuid']
        coord_x = candidate['coordX']
        coord_y = candidate['coordY']
        coord_z = candidate['coordZ']
        
        # Buscar anotações da mesma série
        series_annotations = annotations_df[annotations_df['seriesuid'] == series_uid]
        
        is_negative = True
        min_distance = float('inf')
        
        for _, annotation in series_annotations.iterrows():
            # Calcular distância euclidiana
            distance = np.sqrt(
                (coord_x - annotation['coordX'])**2 +
                (coord_y - annotation['coordY'])**2 +
                (coord_z - annotation['coordZ'])**2
            )
            
            min_distance = min(min_distance, distance)
            
            if distance <= threshold:
                is_negative = False
                break
        
        if is_negative:
            negative_indices.append(idx)
    
    negatives = candidates_df.iloc[negative_indices]
    positives = candidates_df.drop(negative_indices)
    
    print(f"Negativos encontrados: {len(negatives)}")
    print(f"Positivos encontrados: {len(positives)}")
    
    return negatives, positives

def save_negative_list(negatives_df, output_file="negatives_list.json"):
    """Salva lista de negativos em JSON para enviar ao servidor"""
    
    print(f"Salvando lista de negativos em {output_file}...")
    
    # Converter para lista de dicionários
    negatives_list = negatives_df.to_dict('records')
    
    # Salvar JSON
    with open(output_file, 'w') as f:
        json.dump(negatives_list, f, indent=2)
    
    print(f"Lista salva com {len(negatives_list)} negativos")
    
    # Criar resumo
    summary = {
        'total_negatives': len(negatives_list),
        'series_uids': list(negatives_df['seriesuid'].unique()),
        'total_series': len(negatives_df['seriesuid'].unique())
    }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Resumo salvo em {summary_file}")
    return negatives_list

def create_balanced_sample(negatives_df, max_negatives=3255):
    """Cria amostra balanceada de negativos"""
    
    print(f"Criando amostra balanceada com máximo {max_negatives} negativos...")
    
    if len(negatives_df) > max_negatives:
        # Amostrar uniformemente por série
        series_counts = negatives_df['seriesuid'].value_counts()
        samples_per_series = max_negatives // len(series_counts)
        
        balanced_negatives = []
        
        for series_uid in series_counts.index:
            series_negatives = negatives_df[negatives_df['seriesuid'] == series_uid]
            
            if len(series_negatives) <= samples_per_series:
                balanced_negatives.append(series_negatives)
            else:
                # Amostrar aleatoriamente
                sampled = series_negatives.sample(n=samples_per_series, random_state=42)
                balanced_negatives.append(sampled)
        
        balanced_df = pd.concat(balanced_negatives, ignore_index=True)
        
        # Se ainda sobrar espaço, adicionar mais aleatoriamente
        remaining = max_negatives - len(balanced_df)
        if remaining > 0:
            unused_negatives = negatives_df[~negatives_df.index.isin(balanced_df.index)]
            if len(unused_negatives) > 0:
                extra_samples = unused_negatives.sample(n=min(remaining, len(unused_negatives)), random_state=42)
                balanced_df = pd.concat([balanced_df, extra_samples], ignore_index=True)
        
        print(f"Amostra balanceada criada: {len(balanced_df)} negativos")
        return balanced_df
    else:
        print("Todos os negativos serão usados")
        return negatives_df

def main():
    print("IDENTIFICAÇÃO DE CANDIDATOS NEGATIVOS LUNA16")
    print("=" * 50)
    
    # Analisar dados
    result = analyze_candidates_and_annotations()
    if result is None:
        return
    
    candidates_df, annotations_df = result
    
    # Tentar identificar por classe primeiro
    negatives, positives = identify_negatives_by_class(candidates_df)
    
    # Se não tem coluna class, usar coordenadas
    if negatives is None:
        negatives, positives = identify_negatives_by_coordinates(candidates_df, annotations_df)
    
    if negatives is not None and len(negatives) > 0:
        # Criar amostra balanceada
        balanced_negatives = create_balanced_sample(negatives, max_negatives=3255)
        
        # Salvar lista de negativos
        negatives_list = save_negative_list(balanced_negatives, "negatives_for_server.json")
        
        # Estatísticas finais
        print(f"\nEstatísticas finais:")
        print(f"Total de candidatos: {len(candidates_df)}")
        print(f"Negativos selecionados: {len(balanced_negatives)}")
        print(f"Séries únicas nos negativos: {len(balanced_negatives['seriesuid'].unique())}")
        
        print(f"\nArquivos criados:")
        print(f"- negatives_for_server.json: Lista completa de negativos")
        print(f"- negatives_for_server_summary.json: Resumo")
        
        print(f"\nPróximos passos:")
        print(f"1. Enviar estes arquivos JSON para o servidor")
        print(f"2. Usar script no servidor para processar negativos")
        
    else:
        print("Nenhum negativo encontrado!")

if __name__ == "__main__":
    main()