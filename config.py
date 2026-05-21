import os
import torch

# Dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hiperparâmetros de Treinamento
BATCH_SIZE = 8
NUM_EPOCHS = 50  # Aumentado para convergência do detector 
LR = 1e-4        # Valor base do AdamW/AdaBeliefz para Mask R-CNN [cite: 720]

# Configurações do Modelo
NUM_LABELS = 2   # 0 (Fundo) + 1 (Nódulo) [cite: 1516]
IMG_SIZE = 512   # Resolução original do LUNA16 [cite: 1214]

# Configurações de Âncoras (Diferencial para nódulos pequenos < 15mm)
# Isso ajuda a resolver o problema de sensibilidade que você citou no TCC [cite: 1120]
ANCHOR_SIZES = (8, 16, 32, 64, 128) 
ANCHOR_RATIOS = (0.5, 1.0, 2.0)

# Caminhos (Mantendo sua estrutura da UFAL)
TRAIN_FOLDER = "dataset/dataset_10-15mm_train"
TEST_FOLDER = "dataset/dataset_10-15mm_test"
SAVE_MODEL_PATH = "outputs/"

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)