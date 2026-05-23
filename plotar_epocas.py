import os
import glob
import torch
import re
import matplotlib.pyplot as plt

pasta_outputs = 'outputs'
# Pega todos os arquivos .pth da pasta
arquivos_pth = glob.glob(os.path.join(pasta_outputs, '*.pth'))

# Função para ordenar os arquivos corretamente (1, 2, 3... em vez de 1, 10, 100)
def extrair_epoca(nome_arquivo):
    match = re.search(r'epoch_(\d+)', nome_arquivo)
    return int(match.group(1)) if match else 0

arquivos_pth.sort(key=extrair_epoca)

epocas = []
valores_loss = []

print(f"Encontrados {len(arquivos_pth)} arquivos .pth na pasta '{pasta_outputs}'.")
print("Iniciando extração (isso pode levar um minutinho)...")

# --- TESTE DIAGNÓSTICO NO PRIMEIRO ARQUIVO ---
if len(arquivos_pth) > 0:
    chkpt_teste = torch.load(arquivos_pth[0], map_location='cpu')
    if isinstance(chkpt_teste, dict):
        print(f"-> Chaves encontradas no arquivo: {list(chkpt_teste.keys())}")
    else:
        print("-> O .pth contém apenas os pesos puros (não é um dicionário).")

# --- EXTRAÇÃO DE TODOS OS ARQUIVOS ---
for arquivo in arquivos_pth:
    try:
        # map_location='cpu' garante que não vamos estourar a memória da placa de vídeo
        checkpoint = torch.load(arquivo, map_location='cpu')
        
        loss_epoca = None
        if isinstance(checkpoint, dict):
            # Procura pelas chaves de loss mais comuns
            if 'loss' in checkpoint:
                loss_epoca = checkpoint['loss']
            elif 'train_loss' in checkpoint:
                loss_epoca = checkpoint['train_loss']
                
        if loss_epoca is not None:
            if torch.is_tensor(loss_epoca):
                loss_epoca = loss_epoca.item() # Converte tensor para número
                
            epocas.append(extrair_epoca(arquivo))
            valores_loss.append(loss_epoca)
            
    except Exception as e:
        pass # Ignora arquivos corrompidos

# --- PLOTAGEM DO GRÁFICO ---
if len(valores_loss) > 0:
    print(f"\nExtração concluída! Gerando gráfico com {len(epocas)} pontos...")
    plt.figure(figsize=(8, 6))
    plt.plot(epocas, valores_loss, color='#1f77b4', linewidth=2, label='Treino (ConvNeXt V2)')
    
    plt.title('Curva de Aprendizado (Loss)', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('grafico_treinamento_convnext.png', dpi=300)
    print("Sucesso Absoluto! Gráfico salvo como: grafico_treinamento_convnext.png")
else:
    print("\n[AVISO]: O valor de 'loss' não foi salvo dentro dos arquivos .pth.")
    print("Seu script salvou apenas os 'pesos' (model_state_dict).")