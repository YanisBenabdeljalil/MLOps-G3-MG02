from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import os
import joblib
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Optional

# --- CONFIGURATION ---
app = FastAPI(title="MatchMySkills API")

CSV_PATH = "src/data/competency_job.csv"
MODEL_PATH = "models/competency_embeddings.joblib"
MODEL_NAME = "all-MiniLM-L6-v2"

# Variables globales chargées au démarrage
model = None
df = None
comp_unique = None
comp_embs = None # Embeddings chargés depuis le .joblib (ou recalculés si besoin de cohérence absolue)

# Modèle de données reçu depuis Streamlit
class PredictRequest(BaseModel):
    profile: str
    projA: str
    projB: str
    slider_scores: Dict[str, int]  # { "NomCompetence": Score(0-5) }

# Fonctions Cleaning Input User et Logiques Métiers

def clean_user_text(text: str) -> str:
    if not text: return ""
    t = text.strip()
    t = re.sub(r'https?://\S+|www\.\S+', '', t); t = re.sub(r'@[A-Za-z0-9_]+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t); t = re.sub(r'\s+', ' ', t).strip()
    return t

def build_job_requirements(df_in):
    # Grouper les poids par job
    tmp = (df_in[["JobID", "JobTitle", "CompetencyID", "Weight"]]
             .dropna(subset=["CompetencyID", "Weight"])
             .groupby(["JobID", "JobTitle", "CompetencyID"], as_index=False)["Weight"].max())
    job_req = {}
    for (job_id, job_title), g in tmp.groupby(["JobID", "JobTitle"]):
        reqs = list(zip(g["CompetencyID"], g["Weight"]))
        job_req[job_id] = {"title": job_title, "reqs": reqs}
    return job_req

# --- DEMARRAGE DE L'API ---
@app.on_event("startup")
def startup_event():
    global model, df, comp_unique, comp_embs
    print("🚀 Chargement du moteur IA...")
    
    model = SentenceTransformer(MODEL_NAME)
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV introuvable ici : {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Recréation de comp_unique pour la cohérence
    comp_unique = df.groupby("CompetencyID", as_index=False).agg({
        "CompetencyName": lambda s: s.mode().iloc[0] if not s.empty else str(s.iloc[0]),
        "Keywords": lambda s: ", ".join(set(", ".join(s.fillna("").astype(str)).split(","))),
        "Level": lambda s: s.mode().iloc[0] if not s.empty else str(s.iloc[0])
    })
    
    # On recrée la phrase texte pour être sûr de la cohérence
    comp_unique["TextForEmbed"] = (
        comp_unique["CompetencyName"].astype(str) + ". keywords: " + 
        comp_unique["Keywords"].astype(str)
    )

    # 1. Essai de chargement du joblib (plus rapide)
    if os.path.exists(MODEL_PATH):
        print("Chargement des embeddings depuis le disque...")
        comp_embs = joblib.load(MODEL_PATH)
    else:
        # 2. Fallback : Calcul à la volée (Sécurité)
        print("⚠️ .joblib introuvable, calcul des embeddings au démarrage...")
        comp_embs = {
            row.CompetencyID: model.encode(row.TextForEmbed, convert_to_tensor=True)
            for row in comp_unique.itertuples()
        }
    print("✅ API Prête !")


# --- ROUTES ---

@app.get("/health")
def health_check():
    """Renvoie l'état du service ."""
    # On vérifie si le modèle et les données sont bien chargés
    if model is None or df is None:
        return {"status": "degraded", "details": "Model or Data not loaded"}
    return {"status": "ok", "message": "Service is healthy and ready to predict"}

@app.get("/reference-data")
def get_reference_data():
    """Envoie la liste des compétences pour le menu déroulant Streamlit"""
    if comp_unique is not None:
        # On renvoie les noms triés
        return {"skills": sorted(comp_unique["CompetencyName"].unique().tolist())}
    return {"skills": []}

@app.post("/predict")
def predict(request: PredictRequest):
    # 1. TEXTE : Préparation et Encodage
    full_text = f"{request.profile} {request.projA} {request.projB}"
    user_text = clean_user_text(full_text)
    user_vec = model.encode(user_text, convert_to_tensor=True)

    # 2. JOB LEVEL SEMANTIC MATCHING 
    # On reconstruit le texte global par job
    job_texts = (
        df.groupby(["JobID", "JobTitle"]).apply(
            lambda g: "; ".join(
                (g["CompetencyName"].astype(str).str.strip() + 
                ". keywords: " + g["Keywords"].astype(str).str.strip())
            )
        ).reset_index(name="JobText")
    )
    job_texts["JobText"] = job_texts["JobTitle"] + ". " + job_texts["JobText"]
    
    # Calcul de similarité globale User <-> Job Text
    job_rows = []
    for _, row in job_texts.iterrows():
        j_vec = model.encode(row["JobText"], convert_to_tensor=True)
        sim = util.cos_sim(user_vec, j_vec).item()
        job_rows.append({"JobID": row["JobID"], "JobLevelSim": max(0.0, (sim + 1)/2)})
    job_level_df = pd.DataFrame(job_rows)

    # 3. SKILL LEVEL SEMANTIC MATCHING
    job_req = build_job_requirements(df)
    sres = []
    for job_id, meta in job_req.items():
        sims, weights = [], []
        for cid, w in meta["reqs"]:
            if cid in comp_embs:
                sim = util.cos_sim(user_vec, comp_embs[cid]).item()
                sims.append(sim); weights.append(float(w))
        
        score = float(np.dot(sims, weights) / sum(weights)) if sims and sum(weights) > 0 else 0.0
        sres.append({"JobID": job_id, "JobTitle": meta["title"], "SemanticScore": max(0.0, (score + 1)/2)})
    
    sres_df = pd.DataFrame(sres)
    
    # Fusion des deux scores sémantiques (50/50)
    sres_df = sres_df.merge(job_level_df, on="JobID", how="left")
    sres_df["SemanticScore"] = 0.5 * sres_df["SemanticScore"] + 0.5 * sres_df["JobLevelSim"]

    # 4. QUESTIONNAIRE (SLIDERS) MATCHING
    # C'est la logique complexe qui compare le slider à l'embedding le plus proche
    qres_df = pd.DataFrame(columns=["JobID", "QuestionScore"])
    
    if request.slider_scores:
        # On encode les noms des sliders choisis par l'utilisateur
        user_skill_vecs = {name: model.encode(name, convert_to_tensor=True) 
                           for name in request.slider_scores.keys()}
        
        # On cherche le meilleur match dans le CSV pour chaque slider
        matched_scores = {} # {CompetencyID : ScorePondéré}
        
        for csv_id, csv_vec in comp_embs.items():
            best_sim = -1.0
            best_user_name = None
            
            for u_name, u_vec in user_skill_vecs.items():
                sim = util.cos_sim(csv_vec, u_vec).item()
                if sim > best_sim:
                    best_sim = sim
                    best_user_name = u_name
            
            # Normalisation et pondération
            sim01 = max(0.0, (best_sim + 1)/2)
            if best_user_name:
                val_slider = request.slider_scores[best_user_name] / 5.0 # Normalisé 0-1
                matched_scores[csv_id] = val_slider * sim01

        # Calcul du score questionnaire par job
        q_results = []
        for job_id, meta in job_req.items():
            qs, qw = [], []
            for cid, w in meta["reqs"]:
                if cid in matched_scores:
                    qs.append(matched_scores[cid]); qw.append(float(w))
            
            q_score = float(np.dot(qs, qw) / sum(qw)) if qs and sum(qw) > 0 else 0.0
            q_results.append({"JobID": job_id, "QuestionScore": q_score})
        qres_df = pd.DataFrame(q_results)
    
    if qres_df.empty:
         qres_df = pd.DataFrame([{"JobID": j, "QuestionScore": 0.0} for j in job_req.keys()])

    # 5. SCORE GLOBAL FINAL (Alpha 0.6)
    final_df = sres_df.merge(qres_df, on="JobID", how="left").fillna(0)
    final_df["GlobalScore"] = 0.6 * final_df["SemanticScore"] + 0.4 * final_df["QuestionScore"]
    final_df["GlobalScorePct"] = (final_df["GlobalScore"] * 100).round(1)

    # 6. RETOUR
    top_jobs = final_df.sort_values("GlobalScorePct", ascending=False).head(5)
    
    return {
        "top_jobs": top_jobs.to_dict(orient="records"),
    }

@app.get("/metrics")
def get_metrics():
    """Expose les métriques système et ML de base."""
    # Métriques simples pour montrer au jury que tu monitores ton API
    return {
        "system_info": {
            "model_name": MODEL_NAME,
            "library": "sentence-transformers",
            "api_version": "1.0.0"
        },
        "data_stats": {
            # Combien de métiers on a en base ?
            "total_jobs_reference": len(df["JobID"].unique()) if df is not None else 0,
            # Combien de compétences uniques ?
            "total_competencies_reference": len(comp_unique) if comp_unique is not None else 0,
            # Est-ce que les embeddings sont prêts ?
            "embeddings_loaded": len(comp_embs) if comp_embs else 0
        }
    }