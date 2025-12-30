import pandas as pd
import pytest
from src.data.clean_transform import clean_data

def test_clean_data_nominal():
    """
    Test Unitaire : Vérifie que le nettoyage des données fonctionne sur un cas simple.
    Objectif : Valider la logique métier sans dépendre d'AWS.
    """
    # 1. Création d'un petit DataFrame 'sale' (espaces, types mélangés)
    raw_data = {
        "JobID": ["1", "2"],
        "JobTitle": [" Data Scientist ", "DevOps"], # Espace à nettoyer
        "CompetencyID": ["C1", "C2"],
        "CompetencyName": ["NLP", "Docker"],
        "Weight": ["5", "3"], # String à convertir en nombre
        "Level": ["Expert", "Intermediate"],
        "Keywords": ["Python", "Linux"]
    }
    df_raw = pd.DataFrame(raw_data)

    # 2. Exécution de ta fonction
    df_clean = clean_data(df_raw)

    # 3. Vérifications (Assertions)
    # Le titre doit être nettoyé (plus d'espace au début)
    assert df_clean.iloc[0]["JobTitle"] == "Data Scientist"
    # Le poids doit être devenu un chiffre (int ou float)
    assert pd.api.types.is_numeric_dtype(df_clean["Weight"])
    # On vérifie qu'on a bien gardé les 2 lignes
    assert len(df_clean) == 2

def test_clean_data_invalid_weight():
    """Test Unitaire : Vérifie que les poids invalides sont supprimés."""
    df_bad = pd.DataFrame({
        "JobID": ["1"], "JobTitle": ["A"], "CompetencyID": ["C1"],
        "CompetencyName": ["N"], "Level": ["L"], "Keywords": ["K"],
        "Weight": ["10"] # 10 est hors limites (0-5)
    })
    
    # La fonction clean_data filtre les poids > 5
    df_result = clean_data(df_bad)
    assert len(df_result) == 0