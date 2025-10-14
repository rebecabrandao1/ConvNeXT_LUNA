"""
Gera um relatório Markdown do treinamento a partir dos logs gerados por train_balanced.py

Saídas esperadas em output_dir:
- training_history.json
- training_history.csv
- training_curves.png (gerado automaticamente no final do treino)

Uso:
  python src/generate_training_report.py --output-dir outputs_balanced
"""

import os
import json
import argparse
from datetime import datetime


def load_history(output_dir: str):
    json_path = os.path.join(output_dir, 'training_history.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def summarize_history(data: dict):
    epochs = data.get('epochs', [])
    if not epochs:
        return {
            'total_epochs': 0,
            'best_val_f1': None,
            'best_epoch': None,
            'final_val_metrics': None
        }
    total_epochs = len(epochs)
    # Melhor F1 de validacao
    best_idx, best_item = max(enumerate(epochs), key=lambda x: x[1]['val']['f1'])
    best_val_f1 = best_item['val']['f1']
    best_epoch = best_item['epoch']
    final_val = epochs[-1]['val']
    return {
        'total_epochs': total_epochs,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'final_val_metrics': final_val
    }


def write_markdown_report(output_dir: str, summary: dict, data: dict):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'training_report.md')
    curves_rel = 'training_curves.png'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cfg = data.get('config', {})
    ds = data.get('dataset_stats', {})

    lines = []
    lines.append(f"# Relatório de Treinamento\n")
    lines.append(f"Gerado em: {timestamp}\n\n")
    lines.append("## Configuração\n")
    for k, v in cfg.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")
    if ds:
        lines.append("## Dados\n")
        for k, v in ds.items():
            lines.append(f"- {k}: {v}\n")
        lines.append("\n")

    lines.append("## Resumo\n")
    lines.append(f"- Épocas executadas: {summary['total_epochs']}\n")
    if summary['best_val_f1'] is not None:
        lines.append(f"- Melhor Val F1: {summary['best_val_f1']:.4f} (época {summary['best_epoch']})\n")
    if summary['final_val_metrics'] is not None:
        fm = summary['final_val_metrics']
        lines.append(f"- Métricas finais (Val): Loss={fm['loss']:.4f}, Acc={fm['accuracy']:.4f}, F1={fm['f1']:.4f}\n")
    lines.append("\n")

    if os.path.exists(os.path.join(output_dir, curves_rel)):
        lines.append("## Curvas de Treinamento\n")
        lines.append(f"![Curvas]({curves_rel})\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Gerar relatório do treinamento')
    parser.add_argument('--output-dir', default='outputs_balanced', help='Diretorio de saida do treinamento')
    args = parser.parse_args()

    data = load_history(args.output_dir)
    summary = summarize_history(data)
    report_path = write_markdown_report(args.output_dir, summary, data)
    print(f"Relatório gerado: {report_path}")


if __name__ == '__main__':
    main()
