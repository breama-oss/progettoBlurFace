"""
Modulo per la creazione e configurazione del FaceDetectorYN di OpenCV.

Permette di inizializzare il detector e, se configurato, attivare la GPU.
"""

import cv2
from config import MODEL_PATH, USE_GPU

def init_detector(input_size):
    """
    Inizializza un oggetto FaceDetectorYN con parametri predefiniti e opzionale supporto GPU.

    Parametri
    ----------
    input_size : tuple di int
        Dimensione di input (width, height) per il detector. 
        Normalmente ridimensiona i frame prima del rilevamento.

    Returns
    -------
    cv2.FaceDetectorYN
        Oggetto detector pronto all'uso per rilevare volti nei frame.

    Notes
    -----
    - I parametri del detector sono:
        - score_threshold = 0.4
        - nms_threshold = 0.4
        - top_k = 5000
    - Se USE_GPU è True, prova ad attivare il backend CUDA. Se non disponibile, ricade su CPU.
    - Stampa un messaggio di conferma sull'uso della GPU o della CPU.
    """
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
