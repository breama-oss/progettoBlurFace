# Lettura file in cartella input, processamento e salvataggio in cartella output

import glob
import os
from processing.video_processor import process_video
from config import INPUT_FOLDER, OUTPUT_FOLDER

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = []
for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

if not files:
    print("Nessun video trovato.")
    exit()

for video in files:
    name = os.path.splitext(os.path.basename(video))[0]
    output = os.path.join(OUTPUT_FOLDER, f"{name}_blurred.mp4")
    print(f"Processing {video} -> {output}")
    process_video(video, output)
