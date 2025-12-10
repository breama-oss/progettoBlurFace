"""
Strumenti per il calcolo e la manipolazione del box.
"""

import numpy as np

def expand_bbox(x, y, w, h, factor, img_w, img_h):
    """
    Espande il box mantenendo i limiti del frame.

    Args:
        x (int): Coordinata X del box.
        y (int): Coordinata Y del box.
        w (int): Larghezza del box originale.
        h (int): Altezza del box originale.
        factor (float): Fattore di espansione.
        img_w (int): Larghezza dell'immagine.
        img_h (int): Altezza dell'immagine.

    Returns:
        tuple: (x_start, y_start, x_end - x_start, y_end - y_start)
            Il box espanso e limitato all'immagine.
    """
    cx = x + w / 2
    cy = y + h / 2

    desired_w = w * factor 
    desired_h = h * factor

    x_start = max(0, int(cx - desired_w / 2))
    y_start = max(0, int(cy - desired_h / 2))
    x_end = min(img_w, int(cx + desired_w / 2))
    y_end = min(img_h, int(cy + desired_h / 2))

    return x_start, y_start, x_end - x_start, y_end - y_start
