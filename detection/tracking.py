# Logica di tracking e smoothing

from collections import deque
import numpy as np
from config import SMOOTHING, MIN_FACE_SIZE, MAX_DISAPPEARED, DETECTION_SKIP

def create_track(box):
    return {'history': deque([box], maxlen=SMOOTHING), 'disappeared': 0}

def update_tracks(tracks, detected_boxes):
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
