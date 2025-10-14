"""
Script de treinamento para ConvNeXtV2 com dados LUNA16 balanceados
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    ConvNextV2ForImageClassification, 
    ConvNextV2Config,
    AutoImageProcessor,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse

from dataset_balanced import create_balanced_datasets

class BalancedLunaTrainer:
    def __init__(self, config):
        """
        Inicializa o trainer para dados balanceados.
        
        Args:
            config (dict): Configuracoes de treinamento
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Inicializando treinamento balanceado...")
        print(f"   Device: {self.device}")
        print(f"   Dados: {config['data_dir']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Epocas: {config['max_epochs']}")
        
        # Inicializar componentes
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        
        # Historico de treinamento
        self.train_history = []
        self.val_history = []
    
    def _setup_model(self):
        """Configura o modelo ConvNeXtV2."""
        print("Configurando modelo ConvNeXtV2...")
        
        # Configuracao para imagens em escala de cinza
        config = ConvNextV2Config(
            num_channels=1,  # Grayscale
            num_labels=2,    # Binario: com/sem nodulo
            image_size=224
        )
        
        # Modelo
        self.model = ConvNextV2ForImageClassification(config)
        self.model.to(self.device)
        
        # Processador de imagem
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/convnextv2-tiny-1k-224",
            do_rescale=True,
            do_normalize=True
        )
        
        # Contar parametros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Modelo configurado: {trainable_params:,} parametros")
    
    def _setup_data(self):
        """Configura os datasets e dataloaders."""
        print("Carregando datasets balanceados...")
        
        train_dataset, val_dataset, test_dataset = create_balanced_datasets(
            self.config['data_dir'],
            image_processor=self.processor
        )
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        print(f"   Train: {len(train_dataset)} amostras")
        print(f"   Val: {len(val_dataset)} amostras") 
        print(f"   Test: {len(test_dataset)} amostras")
        
        # Salvar stats
        self.dataset_stats = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'train_distribution': train_dataset.get_class_distribution(),
            'val_distribution': val_dataset.get_class_distribution()
        }
    
    def _setup_optimizer(self):
        """Configura otimizador e scheduler."""
        print("Configurando otimizador...")
        
        # Otimizador com weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * self.config['max_epochs']
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function com peso para classes desbalanceadas
        class_weights = self._calculate_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        print(f"   Pesos das classes: {class_weights.cpu().numpy()}")
    
    def _calculate_class_weights(self):
        """Calcula pesos das classes para loss balanceado."""
        # Contar amostras por classe no treino
        sem_nodulo = self.dataset_stats['train_distribution'].get('sem_nodulo', 0)
        com_nodulo = self.dataset_stats['train_distribution'].get('com_nodulo', 0)
        
        total = sem_nodulo + com_nodulo
        
        if total == 0:
            return torch.tensor([1.0, 1.0])
        
        # Peso inversamente proporcional à frequencia
        weight_sem = total / (2.0 * sem_nodulo) if sem_nodulo > 0 else 1.0
        weight_com = total / (2.0 * com_nodulo) if com_nodulo > 0 else 1.0
        
        return torch.tensor([weight_sem, weight_com])
    
    def train_epoch(self, epoch):
        """Treina uma epoca."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoca {epoch + 1}")
        
        for batch in pbar:
            # Mover para device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Metricas
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Atualizar progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calcular metricas da epoca
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self, epoch):
        """Valida o modelo."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Avaliando")
            
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular metricas
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Relatorio detalhado
        if epoch % 10 == 0 or epoch == self.config['max_epochs'] - 1:
            class_names = ['Sem nodulo', 'Com nodulo']
            report = classification_report(
                all_labels, all_preds, 
                target_names=class_names,
                zero_division=0
            )
            print(f"\nRelatorio de classificacao (validacao):")
            print(report)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy, 
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Salva checkpoint do modelo."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'dataset_stats': self.dataset_stats
        }
        
        # Checkpoint regular
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                self.config['output_dir'], 
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
        
        # Melhor modelo
        if is_best:
            best_path = os.path.join(self.config['output_dir'], "best_model.pt")
            torch.save(checkpoint, best_path)
            
            # Salvar configuracao separada
            config_path = os.path.join(self.config['output_dir'], "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def train(self):
        """Loop principal de treinamento."""
        print(f"Iniciando treinamento para {self.config['max_epochs']} epocas...")
        print("=" * 60)
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            print(f"\nEpoca {epoch + 1}/{self.config['max_epochs']}")
            
            # Treinar
            train_metrics = self.train_epoch(epoch)
            
            # Validar
            val_metrics = self.validate(epoch)
            
            # Salvar historico
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Imprimir resultados
            print(f"Resultados epoca {epoch + 1}:")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, " +
                  f"Acc: {train_metrics['accuracy']:.4f}, " +
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"   Val - Loss: {val_metrics['loss']:.4f}, " +
                  f"Acc: {val_metrics['accuracy']:.4f}, " +
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Verificar se melhorou
            current_f1 = val_metrics['f1']
            is_best = current_f1 > best_f1
            
            if is_best:
                best_f1 = current_f1
                patience_counter = 0
                print(f"   Novo melhor F1: {best_f1:.4f}")
            else:
                patience_counter += 1
            
            # Salvar checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping apos {self.config['patience']} epocas sem melhoria")
                break
        
        print(f"Treinamento concluido! Melhor F1: {best_f1:.4f}")
        print(f"Modelos salvos em: {self.config['output_dir']}")
        
        return best_f1

def main():
    """Funcao principal."""
    parser = argparse.ArgumentParser(description='Treinamento ConvNeXtV2 para LUNA16')
    parser.add_argument('--epochs', type=int, default=100, help='Numero de epocas')
    parser.add_argument('--batch-size', type=int, default=4, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Paciencia para early stopping')
    parser.add_argument('--data-dir', default='dataset_balanced', help='Diretorio dos dados')
    parser.add_argument('--output-dir', default='outputs_balanced', help='Diretorio de saida')
    
    args = parser.parse_args()
    
    # Configuracao
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'patience': args.patience,
        'weight_decay': 0.01,
        'num_workers': 4
    }
    
    print("TREINAMENTO LUNA16 - DETECCAO DE NODULOS")
    print("=" * 50)
    print(f"Configuracao:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Verificar dados
    if not os.path.exists(config['data_dir']):
        print(f"ERRO: Diretorio de dados nao encontrado: {config['data_dir']}")
        print("Execute primeiro: python process_luna16.py")
        return
    
    # Treinar
    trainer = BalancedLunaTrainer(config)
    best_f1 = trainer.train()
    
    print(f"\nTreinamento finalizado!")
    print(f"   Melhor F1-Score: {best_f1:.4f}")
    print(f"   Modelos salvos em: {config['output_dir']}")

if __name__ == "__main__":
    main()