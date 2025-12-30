import joblib
from sentence_transformers import util

def verify():
    # 1. Chargement de l'artefact
    model_data = joblib.load("models/competency_embeddings.joblib")
    print(f"Nombre de compétences indexées : {len(model_data)}")

    # 2. Test de cohérence
    # On affiche les IDs des 5 premières compétences pour vérifier
    sample_ids = list(model_data.keys())[:5]
    print(f"Exemples d'IDs de compétences : {sample_ids}")

    if len(model_data) > 0:
        print("✅ Vérification réussie : Le modèle contient des données valides.")
    else:
        print("❌ Erreur : Le modèle est vide.")

if __name__ == "__main__":
    verify()