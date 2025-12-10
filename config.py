"""
Configurazioni e costanti per il sistema di rilevamento volti e anonimizzazione video.

Include:
- Percorsi dei modelli e cartelle
- Parametri di smoothing e blur
- Parametri di rilevamento e tracking
- Opzioni hardware e ottimizzazione
"""

# Percorsi dei file e cartelle
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

# Parametri di smoothing e blur
SMOOTHING = 5
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 50
EXPAND_FACTOR = 1.3

# Parametri di rilevamento
DETECTION_SKIP = 2
RESIZE_DETECTION = 0.75
USE_GPU = True
SCENE_CHANGE_THRESHOLD = 0.3

# Opzioni hardware e sensibilità
ENHANCE_INPUT = False
MIN_FACE_SIZE = 15
MAX_DISAPPEARED = 15
