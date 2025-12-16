# Base image con CUDA e cuDNN
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Imposta la working directory
WORKDIR /app

# Installazione di Python, pip e librerie di sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN python3 -m pip install --upgrade pip

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice
COPY . .

# Variabili d'ambiente
ENV INPUT_FOLDER=/app/input
ENV OUTPUT_FOLDER=/app/output

# Comando di default
CMD ["python3", "main.py"]
