### Para rodar

# 0. Criar ambiente virtual
```bash
python -m venv luna_env

# Windows
luna_env\Scripts\activate

# Linux/Mac  
source luna_env/bin/activate
```

# 1. Instalar dependências
```bash
pip install -r requirements.txt
```

# 2. Configurar dados (SERVIDOR)
```bash
# Opção A: Setup automático
chmod +x setup_data.sh && ./setup_data.sh

# Opção B: Download direto LUNA16 (se disponível)
chmod +x download_luna16.sh && ./download_luna16.sh

# Opção C: Transfer manual
# Transfira seus dados para ~/luna_data/
scp -r dataset_balanced/ usuario@servidor:~/luna_data/
scp -r subsets/ usuario@servidor:~/luna_data/
```

# 3. Processar dados 
```bash 
python process_luna16.py --subset-dir subsets --max-subsets 9
```
# 4. Treinar modelo
```bash
python train_balanced.py --epochs 400 --patience 50
```
# 5. Usar modelo
```bash
python use_model.py --folder dataset_balanced/test
```
# 6. Visualizar inferências
```bash
# Imagem individual
python visualizar.py --image imagem.jpg

# Pasta completa  
python visualizar.py --folder dataset_balanced/test

# Com limite de imagens
python visualizar.py --folder dataset_balanced/test --max-images 50
```

# 7. Gerar gráficos de métricas
```bash
# Gráficos completos de treinamento
python graficos_metricas.py

# Métricas mAP@50 para detecção
python metricas_map50.py
```

## 📊 Estrutura de Dados

### No Desenvolvimento (Local):
```
tcc/
├── dataset_balanced/     # Dados processados
├── subsets/             # LUNA16 raw data  
├── graficos/            # Gráficos gerados
└── ...
```

### No Servidor (Produção):
```
~/luna_data/            # Dados externos ao Git
├── dataset_balanced/   # Link simbólico -> ~/luna_data/dataset_balanced/
├── subsets/           # Link simbólico -> ~/luna_data/subsets/  
└── models/            # Modelos treinados

ConvNeXT_LUNA/         # Repositório Git (só código)
├── setup_data.sh      # Scripts de setup
├── download_luna16.sh # Download automático
└── ...
```