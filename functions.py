## IMPORTING LIBRARIES ##

import pandas as pd
import numpy as np
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer, util


## LOADING AND VALIDATING CSV ##

# We define the required columns that must exist in the CSV file
REQUIRED_COLS = [
    "JobID", "JobTitle", "CompetencyID", "CompetencyName",
    "Weight", "Level", "Keywords"
]

def load_and_validate_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    # We load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # We check if all required columns are present in the CSV
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    # We remove extra spaces and ensure text columns are of string type
    df["JobID"] = df["JobID"].astype(str).str.strip()
    df["JobTitle"] = df["JobTitle"].astype(str).str.strip()
    df["CompetencyID"] = df["CompetencyID"].astype(str).str.strip()
    df["CompetencyName"] = df["CompetencyName"].astype(str).str.strip()
    df["Level"] = df["Level"].astype(str).str.strip()
    df["Keywords"] = df["Keywords"].astype(str).str.strip()

    # We convert the 'Weight' column to a numeric type and validate its range
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    bad_weight = df["Weight"].isna() | (df["Weight"] < 0) | (df["Weight"] > 5)
    assert not bad_weight.any(), "Weights must be numeric in [0,5]. Fix the input."

    return df


## CLEAN AND AGGREGATE TEXT FIELDS ##

# Clean and normalize the user's input text by removing URL, mentions, hashtags and extra spaces
def clean_user_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'@[A-Za-z0-9_]+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Remove duplicate keywords and return a clean list of it 
def dedupe_keywords(series: pd.Series) -> str:
    seen, out = set(), []
    # We join all rows into one string and split them by commas to get individual keywords
    tokens = ", ".join(series.fillna("")).split(",")
    for t in tokens:
        kw = t.strip().lower()
        # We add each unique keyword (case-insensitive) only once
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return ", ".join(out)

# Get the most frequent value (mode) from a text column
def pick_mode(s: pd.Series) -> str:
    s = s.fillna("").astype(str).str.strip()
    m = s.mode()
    return (m.iloc[0] if not m.empty else s.iloc[0]) if len(s) else ""

# Generate a unique list of competencies with clean text for embeddings
def build_competencies_unique(df: pd.DataFrame) -> pd.DataFrame:
    # We group the dataset by CompetencyID to get unique competencies
    competencies_unique = (
        df.groupby("CompetencyID", as_index=False)
          .agg({
              "CompetencyName": pick_mode,     # we take the most common name
              "Keywords": dedupe_keywords,     # we merge and deduplicate all keywords
              "Level": pick_mode               # we take the most common level
          })
    )

    # We create a combined text field used later for sentence embedding
    competencies_unique["TextForEmbed"] = (
        competencies_unique["CompetencyName"].astype(str).str.strip() + ". "
        + "keywords: " + competencies_unique["Keywords"].astype(str).str.strip() + ". "
        + "level: " + competencies_unique["Level"].astype(str).str.strip()
    )
    return competencies_unique


## SBERT MODEL AND EMBEDDINGS ##

# We load the pre-trained SBERT model used for semantic similarity
def load_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

# We build semantic embeddings for each unique competency using the SBERT model
def build_competency_embeddings(competencies_unique: pd.DataFrame, model: SentenceTransformer) -> dict:
    # We encode the text representation of each competency into vector embeddings
    competency_embeddings = {
        row.CompetencyID: model.encode(row.TextForEmbed, convert_to_tensor=True)
        for row in competencies_unique.itertuples(index=False)
    }
    return competency_embeddings


## JOB REQUIREMENTS AND USER EVALUATION ##

# We build a dictionary that maps each job to its required competencies and their weights
def build_job_requirements(df: pd.DataFrame) -> dict:
    # We keep only relevant columns and aggregate by job and competency
    tmp = (df[["JobID", "JobTitle", "CompetencyID", "Weight"]]
             .dropna(subset=["CompetencyID", "Weight"])
             .groupby(["JobID", "JobTitle", "CompetencyID"], as_index=False)["Weight"]
             .max())

    job_req = {}
    # We store each job with its list of (competency, weight) pairs
    for (job_id, job_title), g in tmp.groupby(["JobID", "JobTitle"]):
        reqs = list(zip(g["CompetencyID"].tolist(), g["Weight"].astype(float).tolist()))
        job_req[job_id] = {"title": job_title, "reqs": reqs}
    return job_req


# We evaluate the similarity between a user's profile and all jobs using SBERT embeddings
def evaluate_user(user_text: str, df: pd.DataFrame, comp_embs: dict, model: SentenceTransformer, top_k: int = 3) -> list[dict]:
    # We encode the user's profile text into a semantic vector
    user_vec = model.encode(user_text, convert_to_tensor=True)

    # We get job requirements with their competencies and weights
    job_req = build_job_requirements(df)

    results = []
    # We compare the user vector to each job's competencies
    for job_id, meta in job_req.items():
        sims, weights = [], []
        for comp_id, w in meta["reqs"]:
            emb = comp_embs.get(comp_id)
            if emb is None:
                continue
            # We compute the cosine similarity between the user and each competency
            sim = util.cos_sim(user_vec, emb).item()
            sims.append(sim)
            weights.append(max(0.0, float(w)))

        if sims and sum(weights) > 0:
            # We calculate a weighted average similarity score
            score = float(np.dot(sims, weights) / (sum(weights)))
            n_used = len(sims)
        else:
            score, n_used = 0.0, 0

        results.append({
            "JobID": job_id,
            "JobTitle": meta["title"],
            "Score": score,
            "n_used": n_used
        })

    # We sort jobs by similarity score and return the top matches
    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    return results[:top_k]


## PROCESS AND SCORE THE USER'S QUESTIONNAIRE ANSWERS ##

# We extract the top N most important competencies based on their total weight in the dataset
def top_competencies_for_sliders(df: pd.DataFrame, top_n: int = 8) -> list[tuple[str, str]]:
    # We group by competency and sum their weights across all jobs, then select the top N
    agg = (
        df.groupby(["CompetencyID", "CompetencyName"], as_index=False)["Weight"]
          .sum()
          .sort_values("Weight", ascending=False)
    )
    return list(agg.head(top_n)[["CompetencyID", "CompetencyName"]].itertuples(index=False, name=None))


# We normalize a score value between 0 and 1 (e.g., for user slider inputs)
def normalize_01(x: float, lo: float = 0.0, hi: float = 5.0) -> float:
    x = float(x)
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# We compute a weighted questionnaire score for each job based on the user’s self-rated competencies
def compute_question_scores(df: pd.DataFrame, comp_scores01: dict[str, float]) -> pd.DataFrame:
    # We retrieve the job requirements (competencies + weights)
    job_req = build_job_requirements(df)
    results = []
    # We loop through each job and calculate its weighted score
    for job_id, meta in job_req.items():
        weights, scores = [], []
        for comp_id, w in meta["reqs"]:
            if comp_id in comp_scores01:
                weights.append(float(w))
                scores.append(float(comp_scores01[comp_id]))
        # We compute a weighted average if valid scores exist
        if scores and sum(weights) > 0:
            qscore = float(np.dot(scores, weights) / sum(weights))
            n_used = len(scores)
        else:
            qscore, n_used = 0.0, 0
        results.append({
            "JobID": job_id,
            "JobTitle": meta["title"],
            "QuestionScore": qscore,
            "n_used_q": n_used
        })
    # We return all jobs sorted by their questionnaire score (highest first)
    return pd.DataFrame(results).sort_values("QuestionScore", ascending=False)
