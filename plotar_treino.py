import matplotlib.pyplot as plt

arquivo_log = 'nohup.out' 

epocas = []
train_loss = []

try:
    with open(arquivo_log, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
        
    for linha in linhas:
        # Procura a frase exata que o seu train.py imprimia
        if "Loss Médio da Época:" in linha:
            # Pega apenas o número no final da frase
            valor_loss = float(linha.split(":")[-1].strip())
            train_loss.append(valor_loss)
            epocas.append(len(train_loss))

    if len(train_loss) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(epocas, train_loss, color='#1f77b4', linewidth=2, label='Treino (ConvNeXt V2)')
        
        plt.title('Curva de Aprendizado (Loss)', fontsize=14)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('grafico_treinamento_convnext.png', dpi=300)
        print(f"Sucesso! Gráfico gerado com {len(epocas)} épocas.")
        print("Arquivo salvo como: grafico_treinamento_convnext.png")
    else:
        print("Aviso: Nenhuma linha com 'Loss Médio da Época' foi encontrada no arquivo.")

except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_log}' não foi encontrado na pasta.")