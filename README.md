# Strumento per rilevamento e offuscamento volti 

Uno script Python basato su OpenCV e il modello YuNet per il rilevamento dei volti, ottimizzato per l'offuscamento (con effetto blur) di volti in video con tracciamento, smoothing e rilevamento del cambio scena.

---

## Caratteristiche Principali

* **Elaborazione Batch:** Processa automaticamente tutti i file video presenti nella cartella di input.
* **Rilevamento Veloce:** Utilizza il modello ONNX `FaceDetectorYN` (YuNet) di OpenCV per un'elevata velocità e accuratezza.
* **Supporto GPU (CUDA):** Ottimizzazione delle performance con l'accelerazione CUDA/GPU (se disponibile).
* **Tracciamento (Tracking) con Smoothing:** I box di rilevamento vengono tracciati tra un frame e l'altro, e le posizioni finali sono mediate per un effetto blur più stabile.
* **Offuscamento:** Il box del volto viene espanso per cercare di coprire oltre al viso anche parte della testa e del collo, limitando al contempo le dimensioni massime.
* **Rilevamento Cambio Scena:** Resetta il tracciamento dei volti in caso di un taglio netto nella scena per prevenire incongruenze visive.


---

## Prerequisiti

* Python 3.x

    Per verificare:

    ```bash
    # macOS/Linux
    python3 --version
    
    # Windows
    python --version
    ```

### Setup
1. **Clona il progetto:**
    ```bash
    git clone https://github.com/breama-oss/progettoBlurFace.git
    cd progettoBlurFace
    ```

Il progetto è pre-configurato. Le cartelle `input/`, `output/`, `models/` e il modello **`face_detection_yunet_2023mar.onnx`** sono già presenti.

2.  **Attiva l'Ambiente Virtuale (`venv`):**
    ```bash
    # Per Linux/macOS
    source venv/bin/activate

    # Per Windows (PowerShell)
    venv\Scripts\Activate.ps1
    ```

3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Utilizzo

1.  **Prepara la cartella di input:**
    * Copia tutti i video che vuoi processare all'interno della cartella **`input/`**.

2.  **Esegui lo script:**

    ```bash
    # Per Linux e macOS
    python3 main.py

    # Per Windows
    python main.py
    ```

3.  **Risultato:**
    * I video processati (con il suffisso `_blurred.mp4`) saranno salvati automaticamente nella cartella **`output/`**.

---

## Configurazione (Variabili in `config.py`)

È possibile modificare le impostazioni all'inizio del file `config.py` per adattarle alle proprie esigenze:

| Variabile | Descrizione | Default |
| :--- | :--- | :--- |
| `MODEL_PATH` | Percorso del file ONNX del modello YuNet. | `"models/face_detection_yunet_2023mar.onnx"` |
| `INPUT_FOLDER` | Cartella contenente i video da processare. | `"input/"` |
| `OUTPUT_FOLDER` | Cartella in cui salvare i risultati. | `"output/"` |
| `SMOOTHING` | Numero di frame da usare per la media mobile del box (più alto = più liscio e meno rumore ambientale). | `5` |
| `BLUR_KERNEL` | Dimensione del kernel per l'offuscamento Gaussiano (deve essere dispari). | `(99, 99)` |
| `EXPAND_FACTOR` | Fattore di espansione iniziale del box del volto (es. 1.3 = +30% dimensioni). | `1.3` |
| `DETECTION_SKIP` | Numero di frame da saltare tra una rilevazione completa e l'altra (usa il tracking nel mezzo). | `2` |
| `RESIZE_DETECTION` | Fattore di ridimensionamento del frame prima della rilevazione (più basso = più veloce ma meno accurato). | `0.75` |
| `USE_GPU` | Abilita/Disabilita l'accelerazione GPU se disponibile. | `True` |
| `SCENE_CHANGE_THRESHOLD` | Sensibilità per il rilevamento del cambio scena (0.0-1.0). | `0.3` |
| `MIN_FACE_SIZE` | Dimensione minima (in pixel) di un volto rilevato per essere offuscato. | `15` |
| `MAX_DISAPPEARED` | Numero massimo di frame consecutivi in cui un volto può sparire prima di interrompere il tracciamento. | `15` |
| `ENHANCE_INPUT` | Applica un miglioramento (CLAHE) al frame prima della rilevazione per video a bassa qualità. | `False` |

---

## Funzionamento Ottimizzato

### **1. Rilevamento Frequente vs. Tracciamento (Tracking)**

Per velocizzare l'elaborazione, il rilevamento dei volti con YuNet non avviene su ogni frame, ma viene saltato ogni `DETECTION_SKIP` frame (di default ogni 2 frame). Nei frame saltati, il box del volto viene stimato utilizzando il tracciamento (tracking) basato sulla vicinanza e sull'associazione con i box precedentemente rilevati.

### **2. Smoothing Logaritmico**

La posizione e dimensione finali del box di blur non sono quelle del singolo rilevamento, ma è una **media pesata** degli ultimi box rilevati con `SMOOTHING` . I pesi sono assegnati in modo logaritmico, dando più importanza ai frame più recenti, per cercare di dare una transizione fluida e naturale.

### **3. `expand_bbox`**

La funzione `expand_bbox` oltre a moltiplicare le dimensioni del box effettua due operazioni:
1.  **Limita l'Espansione:** Rende il fattore di espansione meno "aggressivo" se il volto è vicino ai bordi del frame (per non far "saltare" l'area offuscata fuori dallo schermo e cercare di tenere il blur stabile).
2.  **Limiti Assoluti:** Impone un limite massimo di espansione (2.0x) e un limite di copertura del frame (es. max 40% della larghezza) per evitare di offuscare accidentalmente gran parte dello schermo a causa di un rilevamento errato.

---

## Licenza

Questo progetto è distribuito sotto la **Licenza MIT**.

