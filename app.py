import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

# --- CONFIGURATION ---
API_URL = os.environ.get("API_URL", "https://dmbux3nz4f.eu-west-3.awsapprunner.com")

st.set_page_config(page_title="Match My Skills - Cloud Edition", layout="wide")

# --- CSS / DESIGN ---
st.markdown("""
<style>
    .main { padding: 2rem 3rem; background-color: #111827; color: #e5e7eb; }
    h1, h2, h3 { color: #f9fafb; }
    .stTextArea textarea { background-color: #374151; color: white; }
    div[data-testid="stButton"] button { 
        background: linear-gradient(90deg, #3b82f6, #2563eb); 
        color: white; border: none; padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 6])
with col1:
    try: st.image("logo.png", width=100)
    except: st.write("Logo")
with col2:
    st.title("Match My Skills 🚀")
    st.markdown("### Analyse de Profil et Recommandation de Métiers via une API Cloud")

# --- 1. RÉCUPÉRATION DES DONNÉES DE RÉFÉRENCE ---
@st.cache_data
def get_reference_skills():
    """Récupère la liste des compétences depuis l'API pour les sliders."""
    try:
        resp = requests.get(f"{API_URL}/reference-data", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("skills", [])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
    return []

# On charge les compétences dès l'ouverture
skills_list = get_reference_skills()

# --- 2. INTERFACE UTILISATEUR ---

# Section Profil
st.subheader("1. Ton Profil")
user_profile = st.text_area("Description du profil", placeholder="Ex: Data Scientist passionné par le NLP...", height=120)

c1, c2 = st.columns(2)
with c1:
    user_projA = st.text_area("Projet A", placeholder="Description du projet...", height=100)
with c2:
    user_projB = st.text_area("Projet B", placeholder="Description du projet...", height=100)

# Section Compétences
st.subheader("2. Tes Compétences Clés")
if not skills_list:
    st.warning("⚠️ L'API n'a pas renvoyé de compétences. Vérifie ton déploiement.")
else:
    selected_skills = st.multiselect("Sélectionne tes expertises :", options=skills_list)
    
    slider_scores = {}
    if selected_skills:
        cols = st.columns(4)
        for i, skill in enumerate(selected_skills):
            with cols[i % 4]:
                # On stocke le score (0-5)
                slider_scores[skill] = st.slider(skill, 0, 5, 3, key=skill)

    # --- 3. LANCEMENT DE L'ANALYSE ---
    if st.button("Lancer l'Analyse Cloud"):
        if not user_profile and not slider_scores:
            st.warning("Remplis au moins le profil ou une compétence.")
        else:
            with st.spinner("🧠 Le cerveau AWS réfléchit..."):
                # Préparation du payload EXACTEMENT comme attendu par main.py
                payload = {
                    "profile": user_profile,
                    "projA": user_projA,
                    "projB": user_projB,
                    "slider_scores": slider_scores
                }
                
                try:
                    # Appel POST vers l'API
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                    response.raise_for_status() # Lève une erreur si code != 200
                    
                    data = response.json()
                    
                    # --- AFFICHAGE DES RÉSULTATS ---
                    st.success("Analyse terminée avec succès !")
                    
                    # Tableau des Top Jobs
                    df_jobs = pd.DataFrame(data["top_jobs"])
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.markdown("### 🏆 Top 5 Métiers")
                        # Joli affichage du dataframe
                        st.dataframe(
                            df_jobs[["JobTitle", "GlobalScorePct"]].style.format({"GlobalScorePct": "{:.1f} %"}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_res2:
                        st.markdown("### 📊 Visualisation")
                        fig = px.bar(
                            df_jobs, 
                            x="GlobalScorePct", 
                            y="JobTitle", 
                            orientation='h',
                            text="GlobalScorePct",
                            title="Score de compatibilité (%)",
                            color="GlobalScorePct",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                    # Détails (Scores sémantiques vs Sliders)
                    with st.expander("Voir les détails du calcul (Semantic vs Question)"):
                        st.dataframe(df_jobs)

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Erreur de communication avec l'API : {e}")
                except Exception as e:
                    st.error(f"❌ Erreur de traitement : {e}")