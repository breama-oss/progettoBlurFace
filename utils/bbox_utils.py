# Espansione box

import numpy as np

def expand_bbox(x, y, w, h, factor, img_w, img_h):
    cx = x + w / 2
    cy = y + h / 2

    desired_w = w * factor
    desired_h = h * factor

    x_start = max(0, int(cx - desired_w / 2))
    y_start = max(0, int(cy - desired_h / 2))
    x_end = min(img_w, int(cx + desired_w / 2))
    y_end = min(img_h, int(cy + desired_h / 2))

    return x_start, y_start, x_end - x_start, y_end - y_start
