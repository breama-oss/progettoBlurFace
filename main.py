import cv2
import numpy as np
from collections import deque
import time
import os
import glob

# === SETTINGS ===
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

SMOOTHING = 10
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 50
EXPAND_FACTOR = 1.5

# === OTTIMIZZAZIONI ===
DETECTION_SKIP = 3  # aumentato: detection ogni 3 frame
RESIZE_DETECTION = 0.5  # riduci risoluzione per detection
USE_GPU = True
SCENE_CHANGE_THRESHOLD = 0.3

# === LOW QUALITY VIDEO SETTINGS ===
ENHANCE_INPUT = False  # disattivato: riduce CPU del 60%
MIN_FACE_SIZE = 20
MAX_DISAPPEARED = 30  # aumentato per compensare skip

# === DETECTOR ===
detector = cv2.FaceDetectorYN_create(
    model=MODEL_PATH,
    config="",
    input_size=(320, 320),  # ridotto per velocità
    score_threshold=0.5,  # bilanciato
    nms_threshold=0.4,
    top_k=5000
)

if USE_GPU:
    try:
        detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("✓ GPU enabled")
    except:
        print("⚠ GPU not available, using CPU")

# Pre-calcola kernel per sharpening
sharpen_kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])

# === FUNZIONI ===
def euclidean_distance(box1, box2):
    cx1 = box1[0] + box1[2] / 2
    cy1 = box1[1] + box1[3] / 2
    cx2 = box2[0] + box2[2] / 2
    cy2 = box2[1] + box2[3] / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def expand_bbox(x, y, w_box, h_box, factor, img_w, img_h):
    cx = x + w_box / 2
    cy = y + h_box / 2
    
    new_w = int(w_box * factor)
    new_h = int(h_box * factor)
    
    new_x = int(cx - new_w / 2)
    new_y = int(cy - new_h / 2)
    
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, img_w - new_x)
    new_h = min(new_h, img_h - new_y)
    
    return new_x, new_y, new_w, new_h

def apply_blur_optimized(frame, boxes):
    for (x, y, w_box, h_box) in boxes:
        if w_box <= 0 or h_box <= 0:
            continue
            
        roi = frame[y:y+h_box, x:x+w_box]
        
        if roi.size == 0:
            continue
        
        blurred = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
        frame[y:y+h_box, x:x+w_box] = blurred
    
    return frame

def detect_scene_change(prev_frame, curr_frame, threshold=0.3):
    if prev_frame is None:
        return False
    
    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (160, 90)), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(cv2.resize(curr_frame, (160, 90)), cv2.COLOR_BGR2GRAY)
    
    hist_prev = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    hist_curr = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
    
    cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    
    return correlation < (1.0 - threshold)

def enhance_for_detection(frame):
    """Versione LEGGERA per ridurre CPU"""
    # Solo CLAHE, molto più veloce
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

# === TROVA VIDEO ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm']
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

if len(video_files) == 0:
    print(f"⚠ Nessun video trovato in '{INPUT_FOLDER}'")
    exit()

print(f"Trovati {len(video_files)} video da processare:")
for i, video in enumerate(video_files, 1):
    print(f"  {i}. {os.path.basename(video)}")
print()

# === PROCESSO OGNI VIDEO ===
for video_idx, INPUT_VIDEO in enumerate(video_files, 1):
    video_name = os.path.basename(INPUT_VIDEO)
    name_without_ext = os.path.splitext(video_name)[0]
    OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}_blurred.mp4")
    
    print(f"{'='*60}")
    print(f"[{video_idx}/{len(video_files)}] Processing: {video_name}")
    print(f"{'='*60}")
    
    # Apri video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"✗ Errore apertura video: {video_name}\n")
        continue
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detect_w = int(w * RESIZE_DETECTION)
    detect_h = int(h * RESIZE_DETECTION)
    detector.setInputSize((detect_w, detect_h))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    
    print(f"Resolution: {w}x{h} @ {fps}fps")
    print(f"Frames: {total_frames}")
    print(f"Detection: {detect_w}x{detect_h}, skip={DETECTION_SKIP}")
    print()
    
    # Inizializza tracking
    face_tracks = []
    frame_count = 0
    prev_frame = None
    start_time = time.time()
    
    # Processa frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        scene_changed = detect_scene_change(prev_frame, frame, SCENE_CHANGE_THRESHOLD)
        
        if scene_changed:
            face_tracks = []
            run_detection = True
        else:
            run_detection = (frame_count % DETECTION_SKIP == 1)
        
        if run_detection:
            if ENHANCE_INPUT:
                detection_frame = enhance_for_detection(frame)
            else:
                detection_frame = frame
            
            if RESIZE_DETECTION != 1.0:
                small_frame = cv2.resize(detection_frame, (detect_w, detect_h))
            else:
                small_frame = detection_frame
                
            faces = detector.detect(small_frame)
            
            current_boxes = []
            
            if faces[1] is not None:
                for face in faces[1]:
                    if RESIZE_DETECTION != 1.0:
                        x = int(face[0] / RESIZE_DETECTION)
                        y = int(face[1] / RESIZE_DETECTION)
                        w_box = int(face[2] / RESIZE_DETECTION)
                        h_box = int(face[3] / RESIZE_DETECTION)
                    else:
                        x, y, w_box, h_box = map(int, face[:4])
                    
                    if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                        continue
                    
                    x, y, w_box, h_box = expand_bbox(x, y, w_box, h_box, EXPAND_FACTOR, w, h)
                    current_boxes.append([x, y, w_box, h_box])
            
            # Tracking
            if len(face_tracks) == 0 and len(current_boxes) > 0:
                for box in current_boxes:
                    track = {
                        'history': deque(maxlen=SMOOTHING),
                        'disappeared': 0
                    }
                    track['history'].append(box)
                    face_tracks.append(track)
            
            elif len(current_boxes) > 0:
                used_detections = set()
                
                for track in face_tracks:
                    if len(track['history']) == 0:
                        continue
                        
                    last_box = track['history'][-1]
                    min_dist = float('inf')
                    best_idx = -1
                    
                    for i, curr_box in enumerate(current_boxes):
                        if i in used_detections:
                            continue
                        
                        dist = euclidean_distance(last_box, curr_box)
                        
                        if dist < min(w, h) * 0.3 and dist < min_dist:
                            min_dist = dist
                            best_idx = i
                    
                    if best_idx >= 0:
                        track['history'].append(current_boxes[best_idx])
                        track['disappeared'] = 0
                        used_detections.add(best_idx)
                    else:
                        track['disappeared'] += DETECTION_SKIP
                
                for i, box in enumerate(current_boxes):
                    if i not in used_detections:
                        track = {
                            'history': deque(maxlen=SMOOTHING),
                            'disappeared': 0
                        }
                        track['history'].append(box)
                        face_tracks.append(track)
            else:
                for track in face_tracks:
                    track['disappeared'] += DETECTION_SKIP
            
            face_tracks = [t for t in face_tracks if t['disappeared'] < MAX_DISAPPEARED]
        
        else:
            for track in face_tracks:
                track['disappeared'] += 1
        
        # Calcola box smooth e applica blur
        blur_boxes = []
        
        for track in face_tracks:
            if len(track['history']) == 0:
                continue
            
            weights = np.linspace(0.5, 1.0, len(track['history']))
            weights /= weights.sum()
            
            xs = [box[0] for box in track['history']]
            ys = [box[1] for box in track['history']]
            ws = [box[2] for box in track['history']]
            hs = [box[3] for box in track['history']]
            
            smooth_x = int(np.average(xs, weights=weights))
            smooth_y = int(np.average(ys, weights=weights))
            smooth_w = int(np.average(ws, weights=weights))
            smooth_h = int(np.average(hs, weights=weights))
            
            smooth_x = max(0, smooth_x)
            smooth_y = max(0, smooth_y)
            smooth_w = min(smooth_w, w - smooth_x)
            smooth_h = min(smooth_h, h - smooth_y)
            
            if smooth_w > 0 and smooth_h > 0:
                blur_boxes.append((smooth_x, smooth_y, smooth_w, smooth_h))
        
        frame = apply_blur_optimized(frame, blur_boxes)
        writer.write(frame)
        
        # Salva frame ridotto per confronto (riduce memoria)
        if frame_count % 5 == 0:
            prev_frame = frame.copy()
        
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | {frame_count}/{total_frames} | {fps_proc:.1f} fps | ETA: {eta:.1f}s")
    
    cap.release()
    writer.release()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completato in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"✓ Salvato: {OUTPUT_VIDEO}\n")

print(f"\n{'='*60}")
print(f"✓ TUTTI I VIDEO COMPLETATI!")
print(f"✓ {len(video_files)} video processati")
print(f"✓ Output salvati in: {OUTPUT_FOLDER}")
print(f"{'='*60}")