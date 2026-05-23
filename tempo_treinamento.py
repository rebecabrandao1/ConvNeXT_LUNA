import re
from datetime import datetime

arquivo_log = 'nohup.out'

# Procura o padrão de datas que o Python costuma imprimir (ex: 2026-05-22 10:30:00)
padrao_data = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

tempos_inicio_epocas = []

try:
    with open(arquivo_log, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
        
    for linha in linhas:
        # Se a linha mencionar a palavra "Epoch" ou "Época", extraímos a hora dela
        if "Epoch" in linha or "Época" in linha:
            match = re.search(padrao_data, linha)
            if match:
                data_str = match.group(1)
                data_obj = datetime.strptime(data_str, '%Y-%m-%d %H:%M:%S')
                tempos_inicio_epocas.append(data_obj)
                
    if len(tempos_inicio_epocas) >= 2:
        # Calcula a diferença entre todas as épocas consecutivas
        diferencas_segundos = []
        for i in range(1, len(tempos_inicio_epocas)):
            diferenca = (tempos_inicio_epocas[i] - tempos_inicio_epocas[i-1]).total_seconds()
            
            # Ignora tempos absurdamente longos (ex: o treinamento caiu e você reiniciou no outro dia)
            if 0 < diferenca < 3600: 
                diferencas_segundos.append(diferenca)
                
        if diferencas_segundos:
            tempo_medio_seg = sum(diferencas_segundos) / len(diferencas_segundos)
            minutos = int(tempo_medio_seg // 60)
            segundos = int(tempo_medio_seg % 60)
            
            print("\n=============================================")
            print(f" TEMPO REAL (VIA LOG OFICIAL) DO CONVNEXT V2")
            print("=============================================")
            print(f" -> {minutos} minutos e {segundos} segundos por época.")
            print("=============================================\n")
        else:
            print("As épocas foram encontradas, mas a diferença de tempo entre elas é inválida.")
    else:
        print("Não foram encontrados carimbos de data/hora (Timestamps) suficientes nas linhas das épocas.")
        print("Tente abrir o arquivo nohup.out manualmente e ver como ele salva o tempo.")

except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_log}' não foi encontrado na pasta.")