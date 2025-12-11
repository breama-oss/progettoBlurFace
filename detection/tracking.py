"""
Modulo per gestione del tracciamento dei volti rilevati e smoothing dei box.

Le funzioni principali permettono di:
- Creare nuovi track per volti appena rilevati.
- Aggiornare i track esistenti associandoli alle nuove rilevazioni.
- Applicare smoothing alle coordinate dei box per stabilizzare il blur.
"""

from collections import deque
import numpy as np
from config import SMOOTHING, MAX_DISAPPEARED, DETECTION_SKIP

def create_track(box):
    """
    Crea un nuovo track a partire da un box rilevato.

    Parametri
    ----------
    box : tuple
        Box nella forma (x, y, w, h).

    Returns
    -------
    dict
        Dizionario che rappresenta il track, con chiavi:
        - 'history': deque con la storia delle bounding box (maxlen=SMOOTHING)
        - 'disappeared': contatore dei frame in cui il volto non è stato rilevato
    """
    return {'history': deque([box], maxlen=SMOOTHING), 'disappeared': 0}

def update_tracks(tracks, detected_boxes):
    """
    Aggiorna i track esistenti associandoli alle nuove rilevazioni.

    Parametri
    ----------
    tracks : list of dict
        Lista di track correnti.
    detected_boxes : list of tuples
        Lista di bounding box rilevate nel frame corrente.

    Returns
    -------
    list of dict
        Lista aggiornata di track. I track vengono rimossi se 
        il contatore 'disappeared' supera MAX_DISAPPEARED.

    Notes
    -----
    - I track vengono associati alle nuove rilevazioni basandosi
      sulla distanza euclidea tra le coordinate top-left.
    - Se una rilevazione non corrisponde ad alcun track, viene creato un nuovo track.
    - Se un track non viene aggiornato, il suo contatore 'disappeared' aumenta di DETECTION_SKIP.
    """
    if not tracks:
        return [create_track(b) for b in detected_boxes]

    used = set()

    for track in tracks:
        if not track['history']:
            continue
        last_box = track['history'][-1]

        best_idx = -1
        best_dist = float('inf')

        for i, box in enumerate(detected_boxes):
            if i in used:
                continue
            dist = np.linalg.norm(np.array(last_box[:2]) - np.array(box[:2]))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            track['history'].append(detected_boxes[best_idx])
            track['disappeared'] = 0
            used.add(best_idx)
        else:
            track['disappeared'] += DETECTION_SKIP

    for i, box in enumerate(detected_boxes):
        if i not in used:
            tracks.append(create_track(box))

    return [t for t in tracks if t['disappeared'] < MAX_DISAPPEARED]


def smooth_track(track):
    """
    Applica smoothing alle coordinate dei box di un track.

    Parametri
    ----------
    track : dict
        Track contenente la cronologia dei box in 'history'.

    Returns
    -------
    tuple
        Box dopo l'applicazione smooth (x, y, w, h).

    Notes
    -----
    - Il smoothing utilizza una media pesata logaritmica degli ultimi box.
    - Più recenti sono i box, maggiore è il loro peso.
    - Aiuta a stabilizzare il blur e ridurre oscillazioni tra frame consecutivi.
    """
    weights = np.logspace(-2, 0, len(track['history']))
    weights /= weights.sum()

    xs = [b[0] for b in track['history']]
    ys = [b[1] for b in track['history']]
    ws = [b[2] for b in track['history']]
    hs = [b[3] for b in track['history']]

    x = int(np.average(xs, weights=weights))
    y = int(np.average(ys, weights=weights))
    w = int(np.average(ws, weights=weights))
    h = int(np.average(hs, weights=weights))

    return (x, y, w, h)
