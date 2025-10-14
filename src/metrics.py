#!/usr/bin/env python3
"""
🔷 GERADOR DE GRÁFICOS - MÉTRICAS DE TREINAMENTO LUNA16
Cria visualizações completas das métricas: Loss, Accuracy, F1, Precision, Recall, mAP@50
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse

class GraficosMetricas:
    def __init__(self, estilo='seaborn-v0_8'):
        """
        Inicializa o gerador de gráficos.
        
        Args:
            estilo: Estilo dos gráficos matplotlib
        """
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Cores personalizadas
        self.cores = {
            'train': '#2E86AB',      # Azul para treino
            'val': '#A23B72',        # Roxo para validação
            'test': '#F18F01',       # Laranja para teste
            'loss': '#E63946',       # Vermelho para loss
            'accuracy': '#2A9D8F',   # Verde para accuracy
            'f1': '#F77F00',         # Laranja para F1
            'precision': '#FCBF49',  # Amarelo para precision
            'recall': '#7209B7'      # Roxo para recall
        }
        
        print("🎨 Gerador de gráficos inicializado!")
    
    def carregar_historico_treinamento(self, caminho_modelo):
        """
        Carrega histórico do treinamento a partir do checkpoint.
        
        Args:
            caminho_modelo: Caminho para o arquivo .pt do modelo
            
        Returns:
            dict: Histórico de treinamento
        """
        try:
            import torch
            checkpoint = torch.load(caminho_modelo, map_location='cpu')
            
            if 'train_history' in checkpoint:
                return checkpoint['train_history'], checkpoint.get('val_history', [])
            else:
                print("⚠️ Histórico não encontrado no checkpoint")
                return [], []
                
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint: {e}")
            return [], []
    
    def carregar_resultados_teste(self, caminho_json):
        """
        Carrega resultados de teste do arquivo JSON.
        
        Args:
            caminho_json: Caminho para analysis_results.json
            
        Returns:
            dict: Resultados de teste
        """
        try:
            with open(caminho_json, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ Erro ao carregar resultados: {e}")
            return {}
    
    def criar_grafico_treinamento(self, train_history, val_history, salvar_em="graficos"):
        """
        Cria gráfico completo do histórico de treinamento.
        
        Args:
            train_history: Lista com métricas de treino por época
            val_history: Lista com métricas de validação por época
            salvar_em: Diretório para salvar
        """
        if not train_history:
            print("⚠️ Sem dados de treinamento para plotar")
            return
        
        os.makedirs(salvar_em, exist_ok=True)
        
        # Converter para DataFrame
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history) if val_history else pd.DataFrame()
        
        epochs = range(1, len(train_df) + 1)
        
        # Criar subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Histórico de Treinamento - Detecção de Nódulos LUNA16', 
                     fontsize=16, fontweight='bold')
        
        # 1. Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_df['train_loss'], label='Train Loss', 
                color=self.cores['train'], linewidth=2, marker='o', markersize=3)
        if not val_df.empty:
            ax1.plot(epochs, val_df['val_loss'], label='Validation Loss', 
                    color=self.cores['val'], linewidth=2, marker='s', markersize=3)
        
        ax1.set_title('Loss por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_df['train_accuracy'], label='Train Accuracy', 
                color=self.cores['train'], linewidth=2, marker='o', markersize=3)
        if not val_df.empty:
            ax2.plot(epochs, val_df['val_accuracy'], label='Validation Accuracy', 
                    color=self.cores['val'], linewidth=2, marker='s', markersize=3)
        
        ax2.set_title('Acurácia por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. F1-Score
        ax3 = axes[1, 0]
        ax3.plot(epochs, train_df['train_f1'], label='Train F1-Score', 
                color=self.cores['train'], linewidth=2, marker='o', markersize=3)
        if not val_df.empty:
            ax3.plot(epochs, val_df['val_f1'], label='Validation F1-Score', 
                    color=self.cores['val'], linewidth=2, marker='s', markersize=3)
        
        ax3.set_title('F1-Score por Época')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision & Recall
        ax4 = axes[1, 1]
        if 'train_precision' in train_df.columns:
            ax4.plot(epochs, train_df['train_precision'], label='Train Precision', 
                    color=self.cores['precision'], linewidth=2, marker='o', markersize=3)
            ax4.plot(epochs, train_df['train_recall'], label='Train Recall', 
                    color=self.cores['recall'], linewidth=2, marker='^', markersize=3)
        
        if not val_df.empty and 'val_precision' in val_df.columns:
            ax4.plot(epochs, val_df['val_precision'], label='Val Precision', 
                    color=self.cores['precision'], linewidth=2, linestyle='--', 
                    marker='o', markersize=3)
            ax4.plot(epochs, val_df['val_recall'], label='Val Recall', 
                    color=self.cores['recall'], linewidth=2, linestyle='--', 
                    marker='^', markersize=3)
        
        ax4.set_title('Precision & Recall por Época')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho = os.path.join(salvar_em, f"treinamento_{timestamp}.png")
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"📈 Gráfico de treinamento salvo: {caminho}")
        
        plt.show()
        return caminho
    
    def criar_matriz_confusao(self, resultados_teste, salvar_em="graficos"):
        """
        Cria matriz de confusão a partir dos resultados de teste.
        
        Args:
            resultados_teste: Resultados do teste (dict do JSON)
            salvar_em: Diretório para salvar
        """
        if 'results' not in resultados_teste:
            print("⚠️ Resultados de teste não encontrados")
            return
        
        os.makedirs(salvar_em, exist_ok=True)
        
        # Extrair predições e ground truth
        y_true = []
        y_pred = []
        
        for resultado in resultados_teste['results']:
            # Determinar ground truth pela presença de anotações
            tem_nodulo_real = self._tem_nodulo_real(resultado)
            y_true.append(1 if tem_nodulo_real else 0)
            
            # Predição do modelo
            y_pred.append(resultado['predicted_class'])
        
        # Criar matriz de confusão
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotar
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sem Nódulo', 'Com Nódulo'],
                   yticklabels=['Sem Nódulo', 'Com Nódulo'],
                   ax=ax)
        
        ax.set_title('Matriz de Confusão - Teste Final')
        ax.set_xlabel('Predição')
        ax.set_ylabel('Ground Truth')
        
        # Adicionar métricas
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Texto com métricas
        metricas_texto = f"""
        Acurácia: {report['accuracy']:.3f}
        F1-Score: {report['macro avg']['f1-score']:.3f}
        Precision: {report['macro avg']['precision']:.3f}
        Recall: {report['macro avg']['recall']:.3f}
        """
        
        plt.figtext(0.02, 0.02, metricas_texto, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Salvar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho = os.path.join(salvar_em, f"matriz_confusao_{timestamp}.png")
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"🎯 Matriz de confusão salva: {caminho}")
        
        plt.show()
        return caminho
    
    def criar_grafico_distribuicao(self, resultados_teste, salvar_em="graficos"):
        """
        Cria gráfico de distribuição das probabilidades.
        
        Args:
            resultados_teste: Resultados do teste
            salvar_em: Diretório para salvar
        """
        if 'results' not in resultados_teste:
            return
        
        os.makedirs(salvar_em, exist_ok=True)
        
        # Extrair probabilidades
        prob_sem_nodulo = []
        prob_com_nodulo = []
        classes_reais = []
        
        for resultado in resultados_teste['results']:
            prob_sem_nodulo.append(resultado['probabilities']['sem_nodulo'])
            prob_com_nodulo.append(resultado['probabilities']['com_nodulo'])
            
            tem_nodulo_real = self._tem_nodulo_real(resultado)
            classes_reais.append('Com Nódulo' if tem_nodulo_real else 'Sem Nódulo')
        
        # Criar DataFrame
        df = pd.DataFrame({
            'Prob_Sem_Nodulo': prob_sem_nodulo,
            'Prob_Com_Nodulo': prob_com_nodulo,
            'Classe_Real': classes_reais
        })
        
        # Plotar
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Distribuição de Probabilidades', fontsize=14, fontweight='bold')
        
        # Histograma das probabilidades para nódulos
        ax1 = axes[0]
        for classe in df['Classe_Real'].unique():
            subset = df[df['Classe_Real'] == classe]
            ax1.hist(subset['Prob_Com_Nodulo'], alpha=0.7, label=classe, bins=20)
        
        ax1.set_title('Distribuição: Probabilidade de Nódulo')
        ax1.set_xlabel('Probabilidade de Ter Nódulo')
        ax1.set_ylabel('Frequência')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot por classe
        ax2 = axes[1]
        df.boxplot(column='Prob_Com_Nodulo', by='Classe_Real', ax=ax2)
        ax2.set_title('Box Plot: Probabilidades por Classe Real')
        ax2.set_xlabel('Classe Real')
        ax2.set_ylabel('Probabilidade de Ter Nódulo')
        
        plt.tight_layout()
        
        # Salvar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho = os.path.join(salvar_em, f"distribuicao_{timestamp}.png")
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico de distribuição salvo: {caminho}")
        
        plt.show()
        return caminho
    
    def criar_curva_roc(self, resultados_teste, salvar_em="graficos"):
        """
        Cria curva ROC e calcula AUC.
        
        Args:
            resultados_teste: Resultados do teste
            salvar_em: Diretório para salvar
        """
        if 'results' not in resultados_teste:
            return
        
        from sklearn.metrics import roc_curve, auc
        
        os.makedirs(salvar_em, exist_ok=True)
        
        # Preparar dados
        y_true = []
        y_scores = []
        
        for resultado in resultados_teste['results']:
            tem_nodulo_real = self._tem_nodulo_real(resultado)
            y_true.append(1 if tem_nodulo_real else 0)
            y_scores.append(resultado['probabilities']['com_nodulo'])
        
        # Calcular ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plotar
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.plot(fpr, tpr, color=self.cores['f1'], lw=3, 
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
        ax.set_title('Curva ROC - Detecção de Nódulos')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho = os.path.join(salvar_em, f"roc_curve_{timestamp}.png")
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"📈 Curva ROC salva: {caminho} (AUC: {roc_auc:.3f})")
        
        plt.show()
        return caminho, roc_auc
    
    def _tem_nodulo_real(self, resultado):
        """Determina se a imagem realmente tem nódulo baseado no nome do arquivo."""
        # Implementar lógica para determinar ground truth
        # Por exemplo, verificar se existe arquivo .txt correspondente com conteúdo
        return resultado.get('has_nodule', False)  # Placeholder
    
    def gerar_relatorio_completo(self, caminho_modelo, caminho_resultados, salvar_em="graficos"):
        """
        Gera relatório completo com todos os gráficos.
        
        Args:
            caminho_modelo: Caminho para best_model.pt
            caminho_resultados: Caminho para analysis_results.json
            salvar_em: Diretório para salvar
        """
        print("🚀 Gerando relatório completo de métricas...")
        
        # Carregar dados
        train_hist, val_hist = self.carregar_historico_treinamento(caminho_modelo)
        resultados_teste = self.carregar_resultados_teste(caminho_resultados)
        
        graficos_gerados = []
        
        # 1. Gráfico de treinamento
        if train_hist:
            grafico1 = self.criar_grafico_treinamento(train_hist, val_hist, salvar_em)
            graficos_gerados.append(grafico1)
        
        # 2. Matriz de confusão
        if resultados_teste:
            grafico2 = self.criar_matriz_confusao(resultados_teste, salvar_em)
            graficos_gerados.append(grafico2)
            
            # 3. Distribuição de probabilidades
            grafico3 = self.criar_grafico_distribuicao(resultados_teste, salvar_em)
            graficos_gerados.append(grafico3)
            
            # 4. Curva ROC
            grafico4, auc_score = self.criar_curva_roc(resultados_teste, salvar_em)
            graficos_gerados.append(grafico4)
        
        print(f"✅ Relatório completo gerado!")
        print(f"📁 Gráficos salvos em: {salvar_em}/")
        
        return graficos_gerados

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Gerador de gráficos de métricas LUNA16')
    parser.add_argument('--modelo', default='outputs_balanced/best_model.pt',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--resultados', default='analysis_results.json',
                       help='Caminho para resultados de teste')
    parser.add_argument('--output', default='graficos',
                       help='Diretório para salvar gráficos')
    
    args = parser.parse_args()
    
    print("📊 GERADOR DE GRÁFICOS - MÉTRICAS LUNA16")
    print("=" * 50)
    
    # Inicializar gerador
    gerador = GraficosMetricas()
    
    # Verificar arquivos
    if not os.path.exists(args.modelo):
        print(f"⚠️ Modelo não encontrado: {args.modelo}")
    
    if not os.path.exists(args.resultados):
        print(f"⚠️ Resultados não encontrados: {args.resultados}")
    
    # Gerar relatório completo
    graficos = gerador.gerar_relatorio_completo(
        args.modelo, 
        args.resultados, 
        args.output
    )
    
    print(f"\n🎉 Concluído! {len(graficos)} gráficos gerados")
    print("📊 Métricas disponíveis:")
    print("   • Histórico de treinamento (Loss, Accuracy, F1)")
    print("   • Matriz de confusão")
    print("   • Distribuição de probabilidades")
    print("   • Curva ROC com AUC")

if __name__ == "__main__":
    main()