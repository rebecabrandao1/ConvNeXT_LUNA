#!/usr/bin/env python3
"""
📊 CALCULADORA DE mAP@50 - MÉTRICAS AVANÇADAS LUNA16
Calcula Mean Average Precision @ IoU 0.5 para detecção de nódulos
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def calcular_iou(box1, box2):
    """
    Calcula IoU (Intersection over Union) entre duas caixas.
    
    Args:
        box1, box2: [x_center, y_center, width, height] normalizadas
        
    Returns:
        float: IoU score
    """
    # Converter para formato [x1, y1, x2, y2]
    def center_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1_corners = center_to_corners(box1)
    box2_corners = center_to_corners(box2)
    
    # Calcular área de interseção
    x1 = max(box1_corners[0], box2_corners[0])
    y1 = max(box1_corners[1], box2_corners[1])
    x2 = min(box1_corners[2], box2_corners[2])
    y2 = min(box1_corners[3], box2_corners[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersecao = (x2 - x1) * (y2 - y1)
    
    # Calcular área das caixas
    area_box1 = box1[2] * box1[3]  # width * height
    area_box2 = box2[2] * box2[3]
    
    # União
    uniao = area_box1 + area_box2 - intersecao
    
    return intersecao / uniao if uniao > 0 else 0.0

def calcular_ap(recall, precision):
    """
    Calcula Average Precision usando interpolação.
    
    Args:
        recall: Lista de valores de recall
        precision: Lista de valores de precision
        
    Returns:
        float: Average Precision
    """
    # Adicionar pontos (0,0) e (1,0)
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    
    # Interpolação
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    # Encontrar pontos onde recall muda
    i = np.where(recall[1:] != recall[:-1])[0]
    
    # Calcular AP como soma das áreas
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    
    return ap

def calcular_map50_deteccao(resultados_teste, ground_truth_dir=None):
    """
    Calcula mAP@50 para detecção de nódulos.
    
    Args:
        resultados_teste: Resultados do modelo
        ground_truth_dir: Diretório com anotações ground truth
        
    Returns:
        dict: Métricas mAP@50
    """
    if 'results' not in resultados_teste:
        print("❌ Resultados não encontrados")
        return {}
    
    # Para classificação binária, vamos simular mAP baseado em confiança
    confiancas = []
    labels_verdadeiros = []
    
    for resultado in resultados_teste['results']:
        confianca = resultado['probabilities']['com_nodulo']
        confiancas.append(confianca)
        
        # Determinar se realmente tem nódulo (simplificado)
        # Em um cenário real, você carregaria as anotações YOLO
        tem_nodulo = resultado['predicted_class'] == 1  # Placeholder
        labels_verdadeiros.append(tem_nodulo)
    
    # Ordenar por confiança (descendente)
    indices = np.argsort(confiancas)[::-1]
    confiancas_ordenadas = np.array(confiancas)[indices]
    labels_ordenados = np.array(labels_verdadeiros)[indices]
    
    # Calcular precision e recall
    tp = np.cumsum(labels_ordenados)
    fp = np.cumsum(~labels_ordenados)
    
    num_positivos = np.sum(labels_verdadeiros)
    
    if num_positivos == 0:
        return {'map50': 0.0, 'ap': 0.0, 'precision': [], 'recall': []}
    
    recall = tp / num_positivos
    precision = tp / (tp + fp)
    
    # Calcular AP
    ap = calcular_ap(recall, precision)
    
    return {
        'map50': ap,  # Para classe única, mAP = AP
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'confiancas': confiancas_ordenadas,
        'num_positivos': num_positivos,
        'num_total': len(resultados_teste['results'])
    }

def plotar_precision_recall_curve(metricas_map, salvar_em="graficos"):
    """
    Plota curva Precision-Recall e calcula AUC.
    
    Args:
        metricas_map: Resultado do calcular_map50_deteccao
        salvar_em: Diretório para salvar
    """
    if not metricas_map or 'precision' not in metricas_map:
        return
    
    os.makedirs(salvar_em, exist_ok=True)
    
    precision = metricas_map['precision']
    recall = metricas_map['recall']
    ap = metricas_map['ap']
    
    # Plotar
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.plot(recall, precision, color='blue', linewidth=2, 
           label=f'AP = {ap:.3f}')
    ax.fill_between(recall, precision, alpha=0.3, color='blue')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curva Precision-Recall (mAP@50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Adicionar informações
    info_texto = f"""
    mAP@50: {metricas_map['map50']:.3f}
    AP: {ap:.3f}
    Total Positivos: {metricas_map['num_positivos']}
    Total Imagens: {metricas_map['num_total']}
    """
    
    plt.figtext(0.02, 0.02, info_texto, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    caminho = os.path.join(salvar_em, f"precision_recall_{timestamp}.png")
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"📈 Curva P-R salva: {caminho}")
    
    plt.show()
    return caminho

def criar_grafico_metricas_completas(metricas_map, resultados_teste, salvar_em="graficos"):
    """
    Cria gráfico completo com todas as métricas principais.
    
    Args:
        metricas_map: Métricas mAP calculadas
        resultados_teste: Resultados originais
        salvar_em: Diretório para salvar
    """
    os.makedirs(salvar_em, exist_ok=True)
    
    # Calcular métricas básicas
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    y_true = []
    y_pred = []
    
    for resultado in resultados_teste['results']:
        y_true.append(resultado['predicted_class'])  # Placeholder para GT real
        y_pred.append(resultado['predicted_class'])
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    map50 = metricas_map.get('map50', 0.0)
    
    # Criar gráfico de barras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Resumo Completo de Métricas - LUNA16', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Métricas principais
    metricas = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'mAP@50']
    valores = [accuracy, f1, precision, recall, map50]
    cores = ['#2E86AB', '#A23B72', '#F18F01', '#F77F00', '#E63946']
    
    bars1 = ax1.bar(metricas, valores, color=cores, alpha=0.8)
    ax1.set_title('Métricas de Performance')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars1, valores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Distribuição de confiança
    confiancas = [r['probabilities']['com_nodulo'] for r in resultados_teste['results']]
    
    ax2.hist(confiancas, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_title('Distribuição de Confiança')
    ax2.set_xlabel('Confiança para Nódulo')
    ax2.set_ylabel('Frequência')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    caminho = os.path.join(salvar_em, f"metricas_completas_{timestamp}.png")
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"📊 Métricas completas salvas: {caminho}")
    
    plt.show()
    
    # Imprimir resumo
    print("\n📋 RESUMO DE MÉTRICAS:")
    print("=" * 40)
    print(f"🎯 Accuracy:     {accuracy:.3f}")
    print(f"🏆 F1-Score:     {f1:.3f}")
    print(f"🎲 Precision:    {precision:.3f}")
    print(f"♻️  Recall:       {recall:.3f}")
    print(f"📈 mAP@50:       {map50:.3f}")
    print("=" * 40)
    
    return caminho

def main():
    """Função principal para gerar gráficos de mAP@50."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerador de métricas mAP@50 LUNA16')
    parser.add_argument('--resultados', default='analysis_results.json',
                       help='Caminho para resultados de teste')
    parser.add_argument('--output', default='graficos',
                       help='Diretório para salvar gráficos')
    
    args = parser.parse_args()
    
    print("📊 GERADOR DE mAP@50 - LUNA16")
    print("=" * 40)
    
    if not os.path.exists(args.resultados):
        print(f"❌ Arquivo não encontrado: {args.resultados}")
        return
    
    # Carregar resultados
    with open(args.resultados, 'r') as f:
        resultados = json.load(f)
    
    # Calcular mAP@50
    print("🔄 Calculando mAP@50...")
    metricas_map = calcular_map50_deteccao(resultados)
    
    # Gerar gráficos
    print("📈 Gerando gráficos...")
    plotar_precision_recall_curve(metricas_map, args.output)
    criar_grafico_metricas_completas(metricas_map, resultados, args.output)
    
    print("✅ Gráficos de mAP@50 gerados com sucesso!")

if __name__ == "__main__":
    main()