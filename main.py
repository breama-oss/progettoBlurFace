import cv2
import os

# ----------------------
# CONFIG
# ----------------------
input_path = "input/video.mp4"
output_path = "output_blur.mp4"

# ----------------------
# Carica YuNet (OpenCV)
# ----------------------
yunet = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",  # scaricato da opencv_zoo
    config="",
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000,
)

# ----------------------
# Video input
# ----------------------
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise Exception("Errore apertura video.")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

yunet.setInputSize((w, h))

# ----------------------
# Video output
# ----------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# ----------------------
# LOOP video
# ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detection
    faces = yunet.detect(frame)
    if faces[1] is not None:
        for det in faces[1]:
            x, y, w2, h2 = det[:4].astype(int)

            # clipping
            x = max(0, x)
            y = max(0, y)
            w2 = min(w2, w - x)
            h2 = min(h2, h - y)

            roi = frame[y:y + h2, x:x + w2]

            if roi.size > 0:
                # blur super forte
                blurred = cv2.GaussianBlur(roi, (101, 101), 0)
                frame[y:y + h2, x:x + w2] = blurred

    out.write(frame)

cap.release()
out.release()
print("✔ Video generato:", output_path)
