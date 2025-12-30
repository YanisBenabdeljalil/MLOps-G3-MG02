import pandas as pd
import os
from download_data import download_to_s3
from clean_transform import clean_data, build_competencies_unique
from load_final import save_processed_data

# On définit le chemin du CSV par rapport à ce script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "competency_job.csv")

def run_full_pipeline():
    """Orchestre le flux complet : Local -> S3 Raw -> Clean -> S3 Processed."""
    print("--- DÉMARRAGE DU DATA PIPELINE (g3mg02) ---")

    # 1. Upload du brut sur S3
    download_to_s3()

    # 2. Transformation
    print(f"Lecture du fichier : {CSV_PATH}")
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Le fichier CSV est absent de {CSV_PATH}")
            
        df_raw = pd.read_csv(CSV_PATH)
        
        # Nettoyage et Agrégation (fonctions de clean_transform.py)
        df_cleaned = clean_data(df_raw)
        df_final = build_competencies_unique(df_cleaned)
        
        # 3. Sauvegarde du résultat propre sur S3
        save_processed_data(df_final, "competencies_processed.csv")
        
        print("--- PIPELINE TERMINÉ AVEC SUCCÈS ---")
    except Exception as e:
        print(f"ÉCHEC DU PIPELINE : {e}")

if __name__ == "__main__":
    run_full_pipeline()