import os

try:
    # Pega a data da primeira e da última época
    tempo_inicio = os.path.getmtime('outputs/detector_epoch_1.pth')
    tempo_fim = os.path.getmtime('outputs/detector_epoch_400.pth')
    
    # Calcula o tempo total em segundos (usando abs para evitar tempo negativo)
    tempo_total_segundos = abs(tempo_fim - tempo_inicio)
    
    # Se o tempo total for menor que 1 hora, os arquivos perderam a data original
    if tempo_total_segundos < 3600:
        print("❌ O servidor apagou as datas originais dos arquivos quando os movemos.")
        print("As datas atuais são apenas do momento da cópia.")
    else:
        # Divide o tempo total pelas 400 épocas
        tempo_por_epoca = tempo_total_segundos / 400
        minutos = int(tempo_por_epoca // 60)
        segundos = int(tempo_por_epoca % 60)
        
        print("\n=============================================")
        print(f" TEMPO REAL DE TREINAMENTO DO CONVNEXT V2")
        print("=============================================")
        print(f" -> {minutos} minutos e {segundos} segundos por época.")
        print("=============================================\n")

except FileNotFoundError:
    print("Arquivos da época 1 ou 400 não encontrados.")