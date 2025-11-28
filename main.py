import cv2
import numpy as np
from collections import deque
import time
import os
import glob

# === SETTINGS AGGIORNATI ===
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"

SMOOTHING = 5
BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 50
EXPAND_FACTOR = 1.3

# === OTTIMIZZAZIONI ===
DETECTION_SKIP = 2
RESIZE_DETECTION = 0.75
USE_GPU = True
SCENE_CHANGE_THRESHOLD = 0.3

# === LOW QUALITY VIDEO SETTINGS ===
ENHANCE_INPUT = False
MIN_FACE_SIZE = 15
MAX_DISAPPEARED = 15

# === DETECTOR ===
detector = cv2.FaceDetectorYN_create(
    model=MODEL_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.4,
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

sharpen_kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])

# === FUNZIONI AGGIORNATE ===

def euclidean_distance(box1, box2):
    cx1 = box1[0] + box1[2] / 2
    cy1 = box1[1] + box1[3] / 2
    cx2 = box2[0] + box2[2] / 2
    cy2 = box2[1] + box2[3] / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def expand_bbox(x, y, w_orig, h_orig, factor, img_w, img_h):
    """
    Espansione intelligente che limita box giganti ai bordi.
    """
    
    # Calcola centro del box originale
    cx = x + w_orig / 2
    cy = y + h_orig / 2
    
    # Calcola distanza dai bordi (normalizzata 0-1)
    edge_margin = 0.15  # 15% dai bordi
    dist_left = cx / img_w
    dist_right = (img_w - cx) / img_w
    dist_top = cy / img_h
    dist_bottom = (img_h - cy) / img_h
    
    # Trova la distanza minima dal bordo più vicino
    min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
    
    # Riduci il fattore di espansione se vicino ai bordi
    if min_edge_dist < edge_margin:
        # Scala lineare: più vicino al bordo = meno espansione
        edge_factor = min_edge_dist / edge_margin
        # Minimo 1.1x, massimo factor
        adjusted_factor = 1.1 + (factor - 1.1) * edge_factor
    else:
        adjusted_factor = factor
    
    # Calcola dimensioni desiderate
    desired_w = w_orig * adjusted_factor
    desired_h = h_orig * adjusted_factor
    
    # Limita espansione massima assoluta
    max_w_limit = w_orig * 2.0
    max_h_limit = h_orig * 2.0
    
    desired_w = min(desired_w, max_w_limit)
    desired_h = min(desired_h, max_h_limit)
    
    # Ulteriore limite: il box espanso non deve superare una certa % del frame
    max_frame_coverage_w = img_w * 0.4  # massimo 40% larghezza frame
    max_frame_coverage_h = img_h * 0.5  # massimo 50% altezza frame
    
    desired_w = min(desired_w, max_frame_coverage_w)
    desired_h = min(desired_h, max_frame_coverage_h)
    
    # Calcola coordinate
    x_start = int(cx - desired_w / 2)
    y_start = int(cy - desired_h / 2)
    x_end = int(cx + desired_w / 2)
    y_end = int(cy + desired_h / 2)
    
    # Clipping ai bordi
    new_x = max(0, x_start)
    new_y = max(0, y_start)
    new_x_end = min(img_w, x_end)
    new_y_end = min(img_h, y_end)
    
    # Calcola dimensioni finali
    new_w = new_x_end - new_x
    new_h = new_y_end - new_y
    
    # Validazione finale
    if new_w <= 0 or new_h <= 0:
        return x, y, w_orig, h_orig
    
    # Sanity check: se il box espanso è sproporzionatamente grande
    expansion_ratio = (new_w * new_h) / (w_orig * h_orig)
    if expansion_ratio > 4.0:  # se l'area è più di 4x l'originale
        # Riduci proporzionalmente
        scale_down = np.sqrt(4.0 / expansion_ratio)
        new_w = int(new_w * scale_down)
        new_h = int(new_h * scale_down)
        
        # Ricentra
        new_x = int(cx - new_w / 2)
        new_y = int(cy - new_h / 2)
        
        # Re-clipping
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
    
    print(f"Resolution: {w}x{h} @ {fps:.2f}fps")
    print(f"Frames: {total_frames}")
    print(f"Detection: {detect_w}x{detect_h}, skip={DETECTION_SKIP}")
    print()
    
    face_tracks = []
    frame_count = 0
    prev_frame = None
    start_time = time.time()
    
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
                        w_box_orig = int(face[2] / RESIZE_DETECTION)
                        h_box_orig = int(face[3] / RESIZE_DETECTION)
                    else:
                        x, y, w_box_orig, h_box_orig = map(int, face[:4])
                    
                    if w_box_orig < MIN_FACE_SIZE or h_box_orig < MIN_FACE_SIZE:
                        continue
                    
                    # Usa expand_bbox migliorato
                    x, y, w_box, h_box = expand_bbox(x, y, w_box_orig, h_box_orig, EXPAND_FACTOR, w, h)
                    current_boxes.append([x, y, w_box, h_box])
            
            # === TRACKING ===
            
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
                    
                    max_allowable_dist = max(last_box[2] * 0.75, MIN_FACE_SIZE * 1.5)
                    
                    for i, curr_box in enumerate(current_boxes):
                        if i in used_detections:
                            continue
                        
                        dist = euclidean_distance(last_box, curr_box)
                        
                        if dist < max_allowable_dist and dist < min_dist:
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
        
        # === BLUR ===
        blur_boxes = []
        
        for track in face_tracks:
            if len(track['history']) == 0:
                continue
            
            weights = np.logspace(np.log10(0.01), np.log10(1.0), len(track['history']))
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