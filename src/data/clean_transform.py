
import pandas as pd
import re

REQUIRED_COLS = [
    "JobID", "JobTitle", "CompetencyID", "CompetencyName",
    "Weight", "Level", "Keywords"
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et valide le DataFrame brut issu du CSV[cite: 111]."""
    # On vérifie si toutes les colonnes requises sont présentes
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    # Nettoyage des espaces et conversion en chaînes de caractères
    # On traite les colonnes textuelles pour éviter les erreurs d'encodage plus tard
    text_cols = ["JobID", "JobTitle", "CompetencyID", "CompetencyName", "Level", "Keywords"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Validation du 'Weight' (doit être entre 0 et 5)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    # On filtre les lignes avec des poids invalides pour garantir la qualité des données
    df = df.dropna(subset=["Weight"])
    df = df[(df["Weight"] >= 0) & (df["Weight"] <= 5)]

    return df

def dedupe_keywords(series: pd.Series) -> str:
    """Supprime les doublons de mots-clés et normalise le texte."""
    seen, out = set(), []
    # On combine toutes les lignes, on sépare par virgule et on nettoie
    tokens = ", ".join(series.fillna("")).split(",")
    for t in tokens:
        kw = t.strip().lower()
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return ", ".join(out)

def pick_mode(s: pd.Series) -> str:
    """Récupère la valeur la plus fréquente (le mode) d'une colonne textuelle."""
    s = s.fillna("").astype(str).str.strip()
    m = s.mode()
    return (m.iloc[0] if not m.empty else s.iloc[0]) if len(s) else ""

def build_competencies_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Génère une liste unique de compétences avec un texte agrégé pour les futurs embeddings."""
    # On groupe par CompetencyID pour fusionner les informations éparpillées dans le CSV
    competencies_unique = (
        df.groupby("CompetencyID", as_index=False)
          .agg({
              "CompetencyName": pick_mode,     
              "Keywords": dedupe_keywords,     
              "Level": pick_mode               
          })
    )

    # Création du champ 'TextForEmbed' qui sera utilisé par SBERT dans le Model Pipeline
    competencies_unique["TextForEmbed"] = (
        competencies_unique["CompetencyName"].astype(str).str.strip() + ". "
        + "keywords: " + competencies_unique["Keywords"].astype(str).str.strip() + ". "
        + "level: " + competencies_unique["Level"].astype(str).str.strip()
    )
    return competencies_unique