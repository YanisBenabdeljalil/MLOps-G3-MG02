import pandas as pd
import boto3
import joblib
import os
import io
from sentence_transformers import SentenceTransformer

# Configuration
BUCKET_NAME = "s3-g3mg02"
PROCESSED_DATA_KEY = "processed/competencies_processed.csv"
MODEL_PATH = "models/competency_embeddings.joblib"

def train_model():
    print("--- DÉMARRAGE DU MODEL PIPELINE ---")
    s3 = boto3.client('s3')

    # 1. Chargement des données depuis S3
    print(f"Téléchargement des données : {PROCESSED_DATA_KEY}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=PROCESSED_DATA_KEY)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    # 2. Chargement du modèle SBERT
    print("Chargement de SBERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Génération des embeddings (Logique de ton functions.py)
    print("Génération des embeddings pour chaque compétence...")
    # On crée un dictionnaire : CompetencyID -> Vector
    embeddings_dict = {
        row.CompetencyID: model.encode(row.TextForEmbed, convert_to_tensor=True)
        for row in df.itertuples(index=False)
    }

    # 4. Sauvegarde locale de l'artefact
    os.makedirs("models", exist_ok=True)
    joblib.dump(embeddings_dict, MODEL_PATH)
    print(f"Modèle sauvegardé localement dans : {MODEL_PATH}")

    # 5. Export du modèle vers S3 (Model Registry)
    print(f"Envoi du modèle vers s3://{BUCKET_NAME}/models/...")
    s3.upload_file(MODEL_PATH, BUCKET_NAME, "models/competency_embeddings.joblib")
    print("--- MODEL PIPELINE TERMINÉ AVEC SUCCÈS ---")

if __name__ == "__main__":
    train_model()