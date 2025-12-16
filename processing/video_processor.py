# Logica di processamento video

import cv2
from tqdm import tqdm
from detection.detector import init_detector
from detection.tracking import update_tracks, smooth_track
from utils.image_utils import apply_blur, detect_scene_change
from utils.bbox_utils import expand_bbox
from config import *

def process_video(input_path, output_path):
    """
    Elabora un video applicando rilevamento volti, tracking e blur, producendo
    un nuovo file video anonimizzato.

    Parametri
    ----------
    input_path : str
        Percorso al file video di input.
    output_path : str
        Percorso dove salvare il video elaborato.

    Flusso di lavoro
    --------
    1. Apertura del video con OpenCV.
    2. Inizializzazione del detector con dimensione ridotta (per velocità).
    3. Ogni `DETECTION_SKIP` frame:
        - Ridimensionamento del frame.
        - Rilevamento volti.
        - Conversione del box alla dimensione originale.
        - Espansione dei box (`expand_bbox`).
        - Aggiornamento dei oggetti tracciati (`update_tracks`).
    4. Per ogni frame:
        - Smoothing dei box tramite `smooth_track`.
        - Applicazione del blur con `apply_blur`.
        - Scrittura del frame nel video di output.
    5. Ogni 5 frame:
        - Memorizzazione del frame corrente (per eventuale rilevamento cambi scena).

    Notes
    -----
    - La logica del tracciamento evita oscillazioni e migliora la stabilità del blur.
    - Il rilevamento viene eseguito a frame saltati per ridurre il carico computazionale.
    - Integrata una barra di caricamento che tiene traccia del progresso fatto durante l'elaborazione video (Progress bar)
    - Il blur è applicato con la funzione `apply_blur` da `utils.image_utils`.
    - Questo metodo non utilizza ancora `detect_scene_change`, ma il frame precedente
      viene comunque salvato per possibile uso successivo.

    Returns
    -------
    None
        La funzione non restituisce valori, ma salva il video elaborato su disco.

    Raises
    ------
    RuntimeError
        Se il file video di input non può essere aperto.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Errore apertura video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = init_detector(
        (int(w * RESIZE_DETECTION), int(h * RESIZE_DETECTION))
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    tracks = []
    frame_count = 0

    with tqdm(
        total=total_frames,
        desc="Processing",
        unit="frame",
        ncols=80
    ) as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            run_detection = (frame_count % DETECTION_SKIP == 1)

            if run_detection:
                resized = cv2.resize(frame, detector.getInputSize())
                faces = detector.detect(resized)
                boxes = []

                if faces[1] is not None:
                    for f in faces[1]:
                        x = int(f[0] / RESIZE_DETECTION)
                        y = int(f[1] / RESIZE_DETECTION)
                        w_box = int(f[2] / RESIZE_DETECTION)
                        h_box = int(f[3] / RESIZE_DETECTION)

                        boxes.append(
                            expand_bbox(
                                x, y, w_box, h_box,
                                EXPAND_FACTOR, w, h
                            )
                        )

                tracks = update_tracks(tracks, boxes)

            smooth_boxes = [smooth_track(t) for t in tracks]
            frame = apply_blur(frame, smooth_boxes)

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()