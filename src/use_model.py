"""
Sistema de inferencia LUNA16 para servidor - Deteccao de nodulos sem emojis
"""

import os
import torch
import json
from PIL import Image
import numpy as np
from transformers import ConvNextV2ForImageClassification
import torchvision.transforms as transforms
from datetime import datetime
import argparse

class LunaNodeDetector:
    def __init__(self, model_path):
        """
        Inicializa o detector de nodulos.
        
        Args:
            model_path: Caminho para o checkpoint do modelo
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Inicializando detector de nodulos...")
        print(f"   Device: {self.device}")
        
        self._load_model()
        
        print("   Modelo carregado com sucesso!")
    
    def _load_model(self):
        """Carrega o modelo treinado."""
        # Carregar checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Configuracao do modelo
        config = checkpoint.get('config', {})
        
        # Recriar modelo
        from transformers import ConvNextV2Config
        
        model_config = ConvNextV2Config(
            num_channels=1,  # Grayscale
            num_labels=2,    # Binario
            image_size=224
        )
        
        self.model = ConvNextV2ForImageClassification(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Custom transforms para manter 1 canal (grayscale)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalização para 1 canal
        ])
        
        # Informacoes do checkpoint
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'f1_score': checkpoint.get('metrics', {}).get('f1', 'N/A'),
            'accuracy': checkpoint.get('metrics', {}).get('accuracy', 'N/A')
        }
        
        # Contar parametros
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Modelo configurado: {total_params:,} parametros")
        print(f"   Checkpoint carregado:")
        print(f"      Epoca: {self.checkpoint_info['epoch']}")
        print(f"      F1-Score: {self.checkpoint_info['f1_score']}")
        print(f"      Acuracia: {self.checkpoint_info['accuracy']}")
    
    def predict_image(self, image_path):
        """
        Faz predicao em uma imagem.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            dict: Resultado da predicao
        """
        # Carregar e preprocessar imagem (manter 1 canal)
        img = Image.open(image_path).convert('L')  # Grayscale
        
        # Aplicar transforms customizadas (mantém 1 canal)
        pixel_values = self.transform(img).unsqueeze(0)  # Add batch dimension
        pixel_values = pixel_values.to(self.device)
        
        # Predicao
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
        # Probabilidades
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Nomes das classes
        class_names = ['Sem nodulo', 'Com nodulo']
        
        return {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'has_nodule': predicted_class == 1,
            'probabilities': {
                'sem_nodulo': probabilities[0][0].item(),
                'com_nodulo': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, image_paths, batch_size=8):
        """
        Faz predicao em lote de imagens.
        
        Args:
            image_paths: Lista de caminhos para imagens
            batch_size: Tamanho do lote
            
        Returns:
            list: Lista de resultados
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Carregar e processar imagens (manter 1 canal)
            batch_tensors = []
            for path in batch_paths:
                img = Image.open(path).convert('L')  # Grayscale
                tensor = self.transform(img)
                batch_tensors.append(tensor)
            
            # Criar batch tensor
            pixel_values = torch.stack(batch_tensors).to(self.device)
            
            # Predicao
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
            
            # Processar resultados
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            class_names = ['Sem nodulo', 'Com nodulo']
            
            for j, (path, pred_class, probs) in enumerate(zip(batch_paths, predicted_classes, probabilities)):
                confidence = probs[pred_class].item()
                
                result = {
                    'image_path': path,
                    'predicted_class': pred_class.item(),
                    'class_name': class_names[pred_class.item()],
                    'confidence': confidence,
                    'has_nodule': pred_class.item() == 1,
                    'probabilities': {
                        'sem_nodulo': probs[0].item(),
                        'com_nodulo': probs[1].item()
                    }
                }
                results.append(result)
        
        return results
    
    def analyze_folder(self, folder_path, output_file=None):
        """
        Analisa todas as imagens em uma pasta.
        
        Args:
            folder_path: Caminho para a pasta
            output_file: Arquivo para salvar resultados (opcional)
            
        Returns:
            dict: Estatisticas da analise
        """
        print(f"Analisando pasta: {folder_path}")
        
        # Encontrar imagens
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(folder_path, file))
        
        if not image_paths:
            print("Nenhuma imagem encontrada!")
            return {}
        
        print(f"Encontradas {len(image_paths)} imagens")
        
        # Processar em lotes
        batch_size = 8
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"Processando lote {i//batch_size + 1}: {len(batch_paths)} imagens")
            
            batch_results = self.predict_batch(batch_paths, batch_size)
            all_results.extend(batch_results)
        
        # Calcular estatisticas
        total_images = len(all_results)
        with_nodules = sum(1 for r in all_results if r['has_nodule'])
        without_nodules = total_images - with_nodules
        
        stats = {
            'total_images': total_images,
            'with_nodules': with_nodules,
            'without_nodules': without_nodules,
            'nodule_percentage': (with_nodules / total_images * 100) if total_images > 0 else 0,
            'results': all_results
        }
        
        # Salvar resultados
        if output_file is None:
            output_file = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Resultado da Analise:")
        print(f"   Total de imagens: {total_images}")
        print(f"   Com nodulos: {with_nodules} ({stats['nodule_percentage']:.1f}%)")
        print(f"   Sem nodulos: {without_nodules} ({100-stats['nodule_percentage']:.1f}%)")
        print(f"   Resultados salvos em: {output_file}")
        
        # Top imagens com maior probabilidade de nodulo
        sorted_results = sorted(all_results, key=lambda x: x['probabilities']['com_nodulo'], reverse=True)
        
        print(f"\nTop 5 imagens com maior probabilidade de nodulo:")
        for i, result in enumerate(sorted_results[:5]):
            filename = os.path.basename(result['image_path'])
            prob = result['probabilities']['com_nodulo'] * 100
            print(f"   {i+1}. {filename}: {prob:.2f}% probabilidade de nodulo")
        
        stats['report_file'] = output_file
        return stats

def main():
    """Funcao principal - demonstracao."""
    parser = argparse.ArgumentParser(description='Detector de nodulos LUNA16')
    parser.add_argument('--model-path', default='outputs_balanced/best_model.pt',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--image', help='Imagem individual para analisar')
    parser.add_argument('--folder', help='Pasta para analisar')
    parser.add_argument('--output', help='Arquivo de saida para resultados')
    
    args = parser.parse_args()
    
    print("DETECTOR DE NODULOS PULMONARES - LUNA16")
    print("Powered by ConvNeXtV2 + Transformers")
    print("=" * 60)
    
    # Verificar modelo
    if not os.path.exists(args.model_path):
        print(f"ERRO: Modelo nao encontrado: {args.model_path}")
        return
    
    # Inicializar detector
    detector = LunaNodeDetector(args.model_path)
    
    if args.image:
        # Analisar imagem individual
        print(f"\nAnalisando imagem: {args.image}")
        print("-" * 40)
        
        if os.path.exists(args.image):
            result = detector.predict_image(args.image)
            
            print(f"Resultado:")
            print(f"   Predicao: {result['class_name']}")
            print(f"   Confianca: {result['confidence']*100:.2f}%")
            print(f"   Tem nodulo: {'SIM' if result['has_nodule'] else 'NAO'}")
            print(f"   Probabilidades:")
            print(f"      Sem nodulo: {result['probabilities']['sem_nodulo']*100:.2f}%")
            print(f"      Com nodulo: {result['probabilities']['com_nodulo']*100:.2f}%")
        else:
            print(f"ERRO: Imagem nao encontrada: {args.image}")
    
    elif args.folder:
        # Analisar pasta
        if os.path.exists(args.folder):
            detector.analyze_folder(args.folder, args.output)
        else:
            print(f"ERRO: Pasta nao encontrada: {args.folder}")
    
    else:
        # Demonstracao padrao
        print("\nDEMONSTRACAO - DETECTOR DE NODULOS LUNA16")
        print("=" * 60)
        
        # Exemplo de uso
        test_folder = "dataset_balanced/test"
        if os.path.exists(test_folder):
            print(f"\nAnalisando pasta de teste: {test_folder}")
            stats = detector.analyze_folder(test_folder, args.output)
        else:
            print(f"Pasta de teste nao encontrada: {test_folder}")
            print("Para usar:")
            print(f"  python {os.path.basename(__file__)} --image sua_imagem.jpg")
            print(f"  python {os.path.basename(__file__)} --folder sua_pasta/")

if __name__ == "__main__":
    main()