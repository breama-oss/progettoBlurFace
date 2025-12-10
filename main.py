"""
Script per processare tutti i video presenti nella cartella di input
applicando rilevamento volti, tracking e blur, e salvandoli nella cartella di output.

Supporta i formati: .mp4, .avi, .mov, .mkv
"""

import glob
import os
from processing.video_processor import process_video
from config import INPUT_FOLDER, OUTPUT_FOLDER

# Crea la cartella di output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Raccoglie tutti i file video dai formati supportati
files = []
for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

# Controlla se ci sono file da processare
if not files:
    print("Nessun video trovato.")
    exit()

# Processa un video alla volta
for video in files:
    # Costruisce il percorso del file di output con suffisso "_blurred"
    name = os.path.splitext(os.path.basename(video))[0]
    output = os.path.join(OUTPUT_FOLDER, f"{name}_blurred.mp4")
    
    print(f"Processing {video} -> {output}")
    process_video(video, output)
