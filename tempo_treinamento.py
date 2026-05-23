import os
import glob
import re

# Pega todos os arquivos da pasta
arquivos = glob.glob('outputs/*.pth')

# Função para ordenar por época (1, 2, 3...)
def extrair_epoca(nome):
    match = re.search(r'epoch_(\d+)', nome)
    return int(match.group(1)) if match else 0

arquivos.sort(key=extrair_epoca)

tempos_por_epoca = []

# Calcula a diferença de tempo de criação entre os arquivos
for i in range(1, len(arquivos)):
    tempo_anterior = os.path.getmtime(arquivos[i-1])
    tempo_atual = os.path.getmtime(arquivos[i])
    
    # Diferença em segundos
    delta_segundos = tempo_atual - tempo_anterior
    
    # Ignora diferenças muito grandes (caso você tenha pausado o treino e voltado no dia seguinte)
    if delta_segundos < 7200: # Ignora intervalos maiores que 2 horas
        tempos_por_epoca.append(delta_segundos)

if tempos_por_epoca:
    tempo_medio = sum(tempos_por_epoca) / len(tempos_por_epoca)
    minutos = int(tempo_medio // 60)
    segundos = int(tempo_medio % 60)
    
    print("\n=============================================")
    print(f" TEMPO MÉDIO DE TREINAMENTO DO CONVNEXT V2")
    print("=============================================")
    print(f" -> {minutos} minutos e {segundos} segundos por época.")
    print("=============================================\n")
else:
    print("Não foi possível calcular. Verifique se os arquivos estão na pasta 'outputs'.")