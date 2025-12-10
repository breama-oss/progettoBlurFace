"""
Funzione per applicazione blur e cambio scena
"""

import cv2
import numpy as np
from config import BLUR_KERNEL, BLUR_SIGMA

def apply_blur(frame, boxes):
    """
    Applica un effetto di blur (Gaussian Blur) alle regioni specificate all'interno del frame.

    Parametri
    ----------
    frame : numpy.ndarray
        L'immagine (frame video) su cui applicare il blur.
    boxes : liste di tuple
        Lista di box, ciascuna nella forma (x, y, w, h),
        che definisce le regioni da sfocare.

    Return
    -------
    numpy.ndarray
        Il frame con le regioni sfocate.

    Notes
    -----
    - Utilizza `cv2.GaussianBlur` con parametri definiti in config (BLUR_KERNEL, BLUR_SIGMA).
    - Se una ROI è vuota (potrebbe capitare ai margini), viene ignorata.
    """
    for (x, y, w, h) in boxes:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
    return frame

def detect_scene_change(prev, curr, threshold):
    """
    Rileva un cambio di scena confrontando l'istogramma del frame precedente con quello attuale.

    Parametri
    ----------
    prev : numpy.ndarray or None
        Il frame precedente. Se None, la funzione ritorna sempre False.
    curr : numpy.ndarray
        Il frame corrente.
    threshold : float
        Valore tra 0 e 1 che rappresenta la soglia di sensibilità al cambiamento.
        Più è basso, più è facile rilevare un cambio di scena.

    Returns
    -------
    bool
        True se viene rilevato un cambio di scena, False altrimenti.

    Notes
    -----
    - Converte i frame in scala di grigi prima del confronto.
    - Utilizza istogrammi a 32 bin normalizzati.
    - Il confronto avviene con `cv2.HISTCMP_CORREL` (correlazione):
      un valore basso indica che i frame sono diversi.
    - La condizione di cambio scena è: corr < (1 - threshold)
    """
    
    if prev is None:
        return False

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return corr < (1 - threshold)
