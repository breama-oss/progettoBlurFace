# Funzione per applicazione blur e cambio scena

import cv2
import numpy as np
from config import BLUR_KERNEL, BLUR_SIGMA

def apply_blur(frame, boxes):
    for (x, y, w, h) in boxes:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
    return frame

def detect_scene_change(prev, curr, threshold):
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
