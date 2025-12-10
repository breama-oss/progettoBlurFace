# Logica di processamento video

import cv2
from detection.detector import init_detector
from detection.tracking import update_tracks, smooth_track
from utils.image_utils import apply_blur, detect_scene_change
from utils.bbox_utils import expand_bbox
from config import *

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Errore apertura video: {input_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = init_detector((int(w * RESIZE_DETECTION), int(h * RESIZE_DETECTION)))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks = []
    frame_count = 0
    prev_frame = None

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
                        expand_bbox(x, y, w_box, h_box, EXPAND_FACTOR, w, h)
                    )

            tracks = update_tracks(tracks, boxes)

        smooth_boxes = [smooth_track(t) for t in tracks]

        frame = apply_blur(frame, smooth_boxes)
        writer.write(frame)

        if frame_count % 5 == 0:
            prev_frame = frame.copy()

    cap.release()
    writer.release()
