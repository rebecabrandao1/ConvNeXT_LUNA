import os

# Configurações do modelo
BATCH_SIZE = 16
NUM_EPOCHS = 400
LR = 2e-4
NUM_LABELS = 2
NUM_CHANNELS = 1

# Caminhos dos dados
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "Luna")  # Ajuste conforme sua estrutura
TRAIN_FOLDER = os.path.join(DATA_PATH, "train")
TEST_FOLDER = os.path.join(DATA_PATH, "test")

# Configurações de treinamento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL_PATH = os.path.join(BASE_PATH, "checkpoints")

# Criar pasta de checkpoints se não existir
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)