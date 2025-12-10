# Creazione e configurazione di FaceDetectorYunet
# Abilitazione/disabilitazione GPU

import cv2
from config import MODEL_PATH, USE_GPU

def init_detector(input_size):
    detector = cv2.FaceDetectorYN_create(
        model=MODEL_PATH,
        config="",
        input_size=input_size,
        score_threshold=0.4,
        nms_threshold=0.4,
        top_k=5000
    )

    if USE_GPU:
        try:
            detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("✓ GPU attivata")
        except:
            print("⚠ GPU non disponibile, uso CPU")

    return detector
