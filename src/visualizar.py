"""
Visualizador de inferencias para servidor - Gera imagens PNG com deteccoes
Versao otimizada sem GUI interativa, apenas salva visualizacoes
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend nao-interativo para servidor
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from use_model_servidor import LunaNodeDetector
import json
from datetime import datetime
import argparse

class VisualizadorServidor:
    def __init__(self, model_path):
        """
        Inicializa visualizador para servidor.
        
        Args:
            model_path: Caminho para modelo treinado
        """
        self.detector = LunaNodeDetector(model_path)
        
        # Configurar cores
        self.cores = {
            'nodulo_detectado': (255, 255, 0),     # Amarelo para nodulos detectados
            'ground_truth': (0, 255, 0),          # Verde para ground truth
            'sem_nodulo': (0, 255, 0),            # Verde para sem nodulo
            'texto_bg': (0, 0, 0),                # Preto para fundo texto
            'texto_fg': (255, 255, 255)           # Branco para texto
        }
        
        print("Visualizador para servidor inicializado")
    
    def analisar_com_marcacoes(self, caminho_imagem, salvar_em="visualizacoes"):
        """
        Analisa imagem e cria visualizacao com marcacoes.
        
        Args:
            caminho_imagem: Caminho para imagem
            salvar_em: Diretorio para salvar
            
        Returns:
            dict: Resultado + caminho da visualizacao
        """
        print(f"Analisando: {os.path.basename(caminho_imagem)}")
        
        # 1. Fazer predicao
        resultado = self.detector.predict_image(caminho_imagem)
        
        # 2. Carregar ground truth
        ground_truth = self._carregar_ground_truth(caminho_imagem)
        
        # 3. Carregar imagem
        img_cv = cv2.imread(caminho_imagem)
        if img_cv is None:
            # Tentar como PIL se OpenCV falhar
            img_pil = Image.open(caminho_imagem).convert('RGB')
            img_rgb = np.array(img_pil)
        else:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        altura, largura = img_rgb.shape[:2]
        
        # 4. Criar visualizacao
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Deteccao de Nodulos - {os.path.basename(caminho_imagem)}', 
                     fontsize=14, fontweight='bold')
        
        # Imagem original
        axes[0].imshow(img_rgb)
        axes[0].set_title('Imagem Original', fontsize=12)
        axes[0].axis('off')
        
        # Imagem com ground truth
        if ground_truth:
            img_gt = self._desenhar_ground_truth(img_rgb.copy(), ground_truth, largura, altura)
            axes[1].imshow(img_gt)
            axes[1].set_title('Ground Truth (Anotacoes Reais)', fontsize=12)
        else:
            axes[1].imshow(img_rgb)
            axes[1].set_title('Sem Anotacoes Disponiveis', fontsize=12)
        axes[1].axis('off')
        
        # Imagem com predicao
        img_pred = self._desenhar_predicao(img_rgb.copy(), resultado, ground_truth, largura, altura)
        axes[2].imshow(img_pred)
        titulo_pred = f"Predicao: {resultado['class_name']} ({resultado['confidence']*100:.1f}%)"
        axes[2].set_title(titulo_pred, fontsize=12)
        axes[2].axis('off')
        
        # Adicionar informacoes
        info_texto = self._gerar_info(resultado, ground_truth)
        fig.text(0.02, 0.02, info_texto, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Salvar
        os.makedirs(salvar_em, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
        nome_arquivo = f"analise_{nome_base}_{timestamp}.png"
        caminho_saida = os.path.join(salvar_em, nome_arquivo)
        
        plt.savefig(caminho_saida, dpi=200, bbox_inches='tight')
        plt.close()  # Importante: fechar para liberar memoria
        
        print(f"Visualizacao salva: {caminho_saida}")
        
        return {
            'resultado': resultado,
            'ground_truth': ground_truth,
            'visualizacao': caminho_saida
        }
    
    def _carregar_ground_truth(self, caminho_imagem):
        """Carrega anotacoes YOLO se existirem."""
        base_name = os.path.splitext(caminho_imagem)[0]
        arquivo_txt = base_name + ".txt"
        
        anotacoes = []
        if os.path.exists(arquivo_txt):
            with open(arquivo_txt, 'r') as f:
                conteudo = f.read().strip()
            
            if conteudo:  # Se tem conteudo
                for linha in conteudo.split('\n'):
                    if linha.strip():
                        partes = linha.strip().split()
                        if len(partes) == 5:
                            classe, x_center, y_center, width, height = map(float, partes)
                            anotacoes.append({
                                'classe': int(classe),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
        
        return anotacoes
    
    def _desenhar_ground_truth(self, img, anotacoes, largura, altura):
        """Desenha retangulos do ground truth."""
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for i, anotacao in enumerate(anotacoes):
            # Converter coordenadas YOLO para pixels
            x_center = anotacao['x_center'] * largura
            y_center = anotacao['y_center'] * altura
            w = anotacao['width'] * largura
            h = anotacao['height'] * altura
            
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            # Desenhar retangulo verde
            cor_gt = self.cores['ground_truth']
            draw.rectangle([x1, y1, x2, y2], outline=cor_gt, width=3)
            
            # Ponto central
            raio = 4
            draw.ellipse([x_center-raio, y_center-raio, x_center+raio, y_center+raio], 
                        fill=cor_gt)
            
            # Texto
            texto = f"GT#{i+1}"
            draw.text((x1, y1-20), texto, fill=cor_gt, font=font)
        
        return np.array(img_pil)
    
    def _desenhar_predicao(self, img, resultado, ground_truth, largura, altura):
        """Desenha resultado da predicao."""
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_grande = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_grande = font
        
        if resultado['has_nodule']:
            # Se detectou nodulo, usar coordenadas do GT se disponivel
            if ground_truth:
                for anotacao in ground_truth:
                    x_center = anotacao['x_center'] * largura
                    y_center = anotacao['y_center'] * altura
                    w = anotacao['width'] * largura * 1.1
                    h = anotacao['height'] * altura * 1.1
                    
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # Retangulo amarelo
                    cor_pred = self.cores['nodulo_detectado']
                    draw.rectangle([x1, y1, x2, y2], outline=cor_pred, width=4)
                    
                    # X no centro
                    draw.line([x_center-8, y_center-8, x_center+8, y_center+8], 
                             fill=cor_pred, width=3)
                    draw.line([x_center-8, y_center+8, x_center+8, y_center-8], 
                             fill=cor_pred, width=3)
                    
                    # Texto de confianca
                    conf_texto = f"DETECTADO: {resultado['confidence']*100:.1f}%"
                    draw.text((x1, y1-25), conf_texto, fill=cor_pred, font=font_grande)
            
            # Indicador no canto
            draw.text((10, 10), "NODULO ENCONTRADO!", fill=self.cores['nodulo_detectado'], font=font_grande)
        
        else:
            # Se nao detectou
            draw.text((10, 10), "NENHUM NODULO", fill=self.cores['sem_nodulo'], font=font_grande)
            
            # Se perdeu algum GT
            if ground_truth:
                for anotacao in ground_truth:
                    x_center = anotacao['x_center'] * largura
                    y_center = anotacao['y_center'] * altura
                    
                    # Circulo vermelho para perdido
                    draw.ellipse([x_center-12, y_center-12, x_center+12, y_center+12], 
                                outline=(255,0,0), width=3)
                    draw.text((x_center+15, y_center-10), "PERDIDO", fill=(255,0,0), font=font)
        
        return np.array(img_pil)
    
    def _gerar_info(self, resultado, ground_truth):
        """Gera texto informativo."""
        info = f"""ANALISE:
Predicao: {resultado['class_name']}
Confianca: {resultado['confidence']*100:.2f}%
Ground Truth: {len(ground_truth)} nodulos anotados"""
        
        # Status
        tem_gt = len(ground_truth) > 0
        detectou = resultado['has_nodule']
        
        if tem_gt and detectou:
            status = "ACERTO - Detectou corretamente"
        elif tem_gt and not detectou:
            status = "ERRO - Nao detectou (falso negativo)"
        elif not tem_gt and detectou:
            status = "ERRO - Detectou inexistente (falso positivo)"
        else:
            status = "ACERTO - Corretamente sem nodulo"
        
        info += f"\nStatus: {status}"
        return info
    
    def analisar_pasta_completa(self, pasta, output_dir="visualizacoes", max_imagens=None):
        """
        Analisa todas as imagens de uma pasta.
        
        Args:
            pasta: Pasta com imagens
            output_dir: Diretorio para salvar visualizacoes
            max_imagens: Limite de imagens (None = todas)
        """
        print(f"Analisando pasta completa: {pasta}")
        
        # Encontrar imagens
        extensoes = ('.jpg', '.jpeg', '.png', '.bmp')
        imagens = [f for f in os.listdir(pasta) if f.lower().endswith(extensoes)]
        
        if max_imagens:
            imagens = imagens[:max_imagens]
        
        print(f"Encontradas {len(imagens)} imagens para processar")
        
        resultados = []
        
        for i, img_nome in enumerate(imagens):
            caminho_img = os.path.join(pasta, img_nome)
            print(f"[{i+1}/{len(imagens)}] Processando: {img_nome}")
            
            try:
                analise = self.analisar_com_marcacoes(caminho_img, output_dir)
                resultados.append({
                    'imagem': img_nome,
                    'resultado': analise['resultado'],
                    'ground_truth': analise['ground_truth'],
                    'visualizacao': analise['visualizacao']
                })
            except Exception as e:
                print(f"ERRO ao processar {img_nome}: {e}")
        
        # Salvar relatorio
        self._salvar_relatorio(resultados, pasta, output_dir)
        
        print(f"Analise completa finalizada!")
        print(f"Visualizacoes salvas em: {output_dir}")
        
        return resultados
    
    def _salvar_relatorio(self, resultados, pasta_origem, output_dir):
        """Salva relatorio JSON da analise."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arquivo_relatorio = os.path.join(output_dir, f"relatorio_{timestamp}.json")
        
        # Estatisticas
        total = len(resultados)
        detectados = sum(1 for r in resultados if r['resultado']['has_nodule'])
        com_gt = sum(1 for r in resultados if len(r['ground_truth']) > 0)
        acertos = 0
        
        for r in resultados:
            tem_gt = len(r['ground_truth']) > 0
            detectou = r['resultado']['has_nodule']
            if (tem_gt and detectou) or (not tem_gt and not detectou):
                acertos += 1
        
        relatorio = {
            'timestamp': timestamp,
            'pasta_origem': pasta_origem,
            'estatisticas': {
                'total_imagens': total,
                'nodulos_detectados': detectados,
                'imagens_com_gt': com_gt,
                'acertos': acertos,
                'acuracia': (acertos / total * 100) if total > 0 else 0
            },
            'resultados_detalhados': resultados
        }
        
        with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Relatorio salvo: {arquivo_relatorio}")
        print(f"Estatisticas: {acertos}/{total} acertos ({relatorio['estatisticas']['acuracia']:.1f}%)")

def main():
    """Funcao principal."""
    parser = argparse.ArgumentParser(description='Visualizador de inferencias para servidor')
    parser.add_argument('--model', default='outputs_balanced/best_model.pt',
                       help='Caminho para modelo treinado')
    parser.add_argument('--image', help='Imagem individual para analisar')
    parser.add_argument('--folder', help='Pasta para analisar')
    parser.add_argument('--output', default='visualizacoes',
                       help='Diretorio para salvar visualizacoes')
    parser.add_argument('--max-images', type=int,
                       help='Maximo de imagens para processar')
    
    args = parser.parse_args()
    
    print("VISUALIZADOR DE INFERENCIAS - SERVIDOR")
    print("=" * 50)
    
    if not os.path.exists(args.model):
        print(f"ERRO: Modelo nao encontrado: {args.model}")
        return
    
    # Inicializar visualizador
    viz = VisualizadorServidor(args.model)
    
    if args.image and os.path.exists(args.image):
        # Imagem individual
        print(f"Analisando imagem: {args.image}")
        viz.analisar_com_marcacoes(args.image, args.output)
    
    elif args.folder and os.path.exists(args.folder):
        # Pasta completa
        viz.analisar_pasta_completa(args.folder, args.output, args.max_images)
    
    else:
        # Padrao: analisar pasta de teste
        pasta_padrao = "dataset_balanced/test"
        if os.path.exists(pasta_padrao):
            print(f"Analisando pasta padrao: {pasta_padrao}")
            viz.analisar_pasta_completa(pasta_padrao, args.output, args.max_images or 20)
        else:
            print("Nenhuma entrada especificada e pasta padrao nao encontrada")
            print("Uso:")
            print(f"  python {os.path.basename(__file__)} --image imagem.jpg")
            print(f"  python {os.path.basename(__file__)} --folder pasta_imagens/")

if __name__ == "__main__":
    main()