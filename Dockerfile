# Utilisation d'une image Python stable et légère
FROM python:3.10-slim

# Définition du dossier de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système (nécessaires pour certaines librairies ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier de dépendances
COPY requirements.txt .

# OPTIMISATION : On installe d'abord la version CPU de torch pour gagner du temps et de la place
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Installation des autres librairies
RUN pip install --no-cache-dir -r requirements.txt

# Copie des dossiers nécessaires (Source et Modèles)
COPY src/ /app/src/
COPY models/ /app/models/

ENV PYTHONPATH=/app

# Exposition du port 8000 pour FastAPI
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]