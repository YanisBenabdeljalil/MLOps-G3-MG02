## IMPORTING LIBRARIES ##

# We import Streamlit to build the interactive web application
# Pandas for data handling 
# Sentence_transformers utility functions for cosine similarity
import streamlit as st
import pandas as pd
from sentence_transformers import util

import plotly.express as px
import plotly.figure_factory as ff

# We import our custom functions for the analysis
from functions import (
    load_and_validate_csv,
    build_competencies_unique,
    load_sbert_model,
    build_competency_embeddings,
    evaluate_user,
    clean_user_text,
    top_competencies_for_sliders,
    normalize_01,
    compute_question_scores
)

# We configure the Streamlit page and custom the CSS styling
st.set_page_config(page_title="Match My Skills", layout="wide")
st.markdown(
    """
<style>
    .main { padding: 2rem 3rem; background-color: #111827; color: #e5e7eb; }
    h1, h2, h3, h4 { font-weight: 600; color: #f9fafb; }

    /* Textareas */
    .stTextArea textarea {
        background-color: #2f3136 !important;
        color: #f9fafb !important;
        border-radius: 10px;
        border: 1px solid #4b5563;
        padding: 0.8rem;
        font-size: 0.95rem;
    }
    .stTextArea textarea::placeholder { color: #9ca3af; font-style: italic; }

    /* Multiselect box */
    .stMultiselect div[role="listbox"] {
        background-color: #2f3136 !important;
        color: #e5e7eb !important;
    }

    /* Labels and captions */
    .stCaption, .stTextArea label, .stSlider label, label { color: #e5e7eb !important; }

    /* Score box */
    .score-box {
        background-color: #2f3136;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #4b5563;
        margin-top: 0.8rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        color: #e5e7eb;
        border-bottom: 1px solid #4b5563;
        padding-bottom: 0.3rem;
    }

    /* Main button */
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: translateY(-1px);
    }

    /* Logo + title container */
    .logo-title-container { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
    .logo-title-container img { width: 100px; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# Logo + Title + Instruction Paragraph
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=130)
with col2:
    st.markdown(
        """
        <div style='padding-top: 1rem;'>
            <h1 style='margin-bottom: 0.3rem; font-weight: 700; color: #f9fafb;'>
                Match My Skills
            </h1>
            <p style='color: #d1d5db; font-size: 1rem;'>
                Fill in the three sections below to evaluate your profile and discover the jobs that best match your skills.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# We define a cached function to load the dataset, build unique competencies, load SBERT model, and compute embeddings
# we use caching so that expensive operations (like loading the CSV and computing embeddings) are done only once,
# which makes the app faster when the user interacts with it multiple times
@st.cache_resource(show_spinner=True)
def _load_all(csv_path: str = "competency_job.csv", model_name: str = "all-MiniLM-L6-v2"):
    df = load_and_validate_csv(csv_path)
    comp_unique = build_competencies_unique(df)
    model = load_sbert_model(model_name)
    comp_embs = build_competency_embeddings(comp_unique, model)
    return df, comp_unique, model, comp_embs

df, comp_unique, model, comp_embs = _load_all()


# 1) Profile

# We create a section for the user to describe their profile in free text
st.markdown('<div class="section-header">1) Your Profile</div>', unsafe_allow_html=True)
st.caption("Briefly describe your profile (role, tasks, industry, tools, etc.)")

user_raw_profile = st.text_area(
    "Profile description:",
    placeholder="Ex: I'm a Data Analyst working with SQL, Power BI, and Python. I build automated dashboards...",
    height=150,
    max_chars=1000,
)


# 2) Projects

# We create a section for the user to describe recent projects they worked on
st.markdown('<div class="section-header">2) Your Recent Projects</div>', unsafe_allow_html=True)
st.caption("Describe one or two recent projects (objectives, data, tools used, results achieved).")

# We display two text areas side by side for Project A and Project B
colA, colB = st.columns(2)
with colA:
    user_raw_projA = st.text_area(
        "Project A",
        placeholder="Ex: Built a sales forecasting model using Python and scikit-learn...",
        height=140,
    )
with colB:
    user_raw_projB = st.text_area(
        "Project B",
        placeholder="Ex: Developed a Power BI dashboard to monitor logistics KPIs...",
        height=140,
    )


# 3) Key Skills

# We create a section for the user to select their main skills and indicate their proficiency level
st.markdown('<div class="section-header">3) Key Skills</div>', unsafe_allow_html=True)
st.caption("Select your main skills and indicate your level (0 = beginner, 5 = expert).")

# We dynamically load all unique competencies from the dataset
custom_skills = [(row.CompetencyID, row.CompetencyName) for row in comp_unique.itertuples(index=False)]
labels_map = {cid: name for cid, name in custom_skills}
choices = [name for _, name in custom_skills]

# We allow the user to select multiple skills
selected = st.multiselect("Select skills you master:", options=choices)
selected_ids = [cid for cid, name in custom_skills if name in selected]

# We create sliders for the user to rate their level for each selected skill
slider_scores = {}
if selected_ids:
    st.caption("Rate your level for each skill (0 = beginner, 5 = expert):")
    cols = st.columns(4)
    for i, cid in enumerate(selected_ids):
        with cols[i % 4]:
            slider_scores[cid] = st.slider(labels_map[cid], 0, 5, 3, key=f"slider_{cid}")


# Evaluation button

# We create a button that triggers the evaluation of the user's profile, projects, and selected skills
if st.button("Evaluate My Complete Profile"):
    combined_text = "\n".join([
        user_raw_profile.strip(),
        user_raw_projA.strip(),
        user_raw_projB.strip(),
    ]).strip()

    user_text = clean_user_text(combined_text)
    if not user_text and not slider_scores:
        st.warning("Please enter some text and/or select skills before evaluating.")
        st.stop()

    # 1) Semantic Evaluation (skill by skill) 
    # We evaluate the user's combined text against job competencies using embeddings
    sres = evaluate_user(user_text, df, comp_embs, model, top_k=50)
    sres_df = pd.DataFrame(sres).rename(columns={"Score": "SemanticScore", "n_used": "n_used_s"})

    # a) We rescale cosine similarity scores from [-1,1] to [0,1] for consistency
    sres_df["SemanticScore"] = ((sres_df["SemanticScore"] + 1.0) / 2.0).clip(0, 1)

    # b) We build a global job-level text by aggregating all competencies per job
    job_texts = (
        df.groupby(["JobID", "JobTitle"]).apply(
            lambda g: "; ".join(
                (g["CompetencyName"].astype(str).str.strip() + 
                ". keywords: " + g["Keywords"].astype(str).str.strip())
            )
        )
        .reset_index(name="JobText")
    )
    job_texts["JobText"] = job_texts["JobTitle"] + ". " + job_texts["JobText"].fillna("")

    # We encode the full user text and each job's aggregated text for comparison
    user_vec_full = model.encode(user_text, convert_to_tensor=True)
    job_text_vecs = {row.JobID: model.encode(row.JobText, convert_to_tensor=True) for _, row in job_texts.iterrows()}

    # We compute job-level similarity and normalize it to [0,1]
    rows = []
    for jid, jvec in job_text_vecs.items():
        sim = util.cos_sim(user_vec_full, jvec).item()      # [-1,1]
        sim01 = max(0.0, (sim + 1.0) / 2.0)                 # -> [0,1]
        rows.append({"JobID": jid, "JobLevelSim": sim01})
    joblevel_df = pd.DataFrame(rows).merge(job_texts[["JobID", "JobTitle"]], on="JobID", how="left")

    # c) We combine skill-by-skill and job-level semantic scores equally (50/50)
    sres_df = sres_df.merge(joblevel_df, on=["JobID", "JobTitle"], how="left")
    sres_df["SemanticScore"] = 0.5 * sres_df["SemanticScore"].fillna(0) + 0.5 * sres_df["JobLevelSim"].fillna(0)
    sres_df.drop(columns=["JobLevelSim"], inplace=True)


    # 2) Questionnaire Evaluation (Skills sliders) 
    if slider_scores:
        comp_scores_01 = {cid: normalize_01(val, 0, 5) for cid, val in slider_scores.items()}

        # We encode user-selected skill labels into embeddings
        user_vecs = {cid: model.encode(labels_map[cid], convert_to_tensor=True) for cid in comp_scores_01.keys()}

        # We encode the full reference text for each competency from the dataset
        comp_csv_vecs = {row.CompetencyID: model.encode(row.TextForEmbed, convert_to_tensor=True)
                        for _, row in comp_unique.iterrows()}

        # We match each dataset competency to the user skill with the highest semantic similarity
        # and weight the user score by the similarity
        matched_scores = {}
        for csv_id, csv_vec in comp_csv_vecs.items():
            best_sim = -1.0
            best_user_cid = None
            for user_cid, user_vec in user_vecs.items():
                sim = util.cos_sim(csv_vec, user_vec).item()  # similarity in [-1,1]
                if sim > best_sim:
                    best_sim = sim
                    best_user_cid = user_cid
            sim01 = max(0.0, (best_sim + 1.0) / 2.0)
            if best_user_cid is not None:
                matched_scores[csv_id] = comp_scores_01[best_user_cid] * sim01

        # we compute the weighted questionnaire score per job
        qres_df = compute_question_scores(df, matched_scores)
    else:
        qres_df = pd.DataFrame(columns=["JobID", "JobTitle", "QuestionScore", "n_used_q"])


    # 3) Merge Semantic & Questionnaire Scores into Global Score 
    alpha = 0.6  # We give 60% weight to semantic score and 40% to questionnaire score

    # We merge the semantic evaluation and questionnaire evaluation for each job
    job_scores_df = pd.merge(sres_df, qres_df, on=["JobID", "JobTitle"], how="outer")

    for col in ["SemanticScore", "QuestionScore"]:
        if col not in job_scores_df:
            job_scores_df[col] = 0.0

    # We compute a global score as a weighted average of semantic and questionnaire scores
    job_scores_df["GlobalScore"] = (
        alpha * job_scores_df["SemanticScore"].fillna(0)
        + (1 - alpha) * job_scores_df["QuestionScore"].fillna(0)
    )

    # We convert the global score to a percentage [0-100] for display
    # And we get the top 5 jobs based on the global score
    job_scores_df["GlobalScorePct"] = (100.0 * job_scores_df["GlobalScore"].clip(0, 1)).round(1)
    top_jobs = job_scores_df.sort_values("GlobalScore", ascending=False).head(5)


    # 4) Display Results
    st.markdown("### Top 5 Matching Jobs")
    # we show the top 5 jobs based on the global score
    st.dataframe(
        top_jobs[["JobTitle", "GlobalScorePct"]]
        .rename(columns={"GlobalScorePct": "GlobalScore (%)"})
        .round(1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Detailed Scores")
    c1, c2 = st.columns(2)
    with c1:
        # We show top semantic scores per job
        st.write("**Semantic Scores (text analysis)**")
        st.dataframe(
            sres_df[["JobTitle", "SemanticScore"]].sort_values("SemanticScore", ascending=False).head(5),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        # We show top questionnaire scores per job
        st.write("**Questionnaire Scores (skills)**")
        st.dataframe(
            qres_df[["JobTitle", "QuestionScore"]].sort_values("QuestionScore", ascending=False).head(5),
            use_container_width=True,
            hide_index=True,
        )

    # 5) Visualization: Global Score Bar Chart 
    st.markdown("### Global Score")
    # We create a horizontal bar chart for top jobs based on global score
    fig = px.bar(
        top_jobs,
        x="JobTitle",
        y="GlobalScorePct",
        text=top_jobs["GlobalScorePct"].round(1),
        labels={"GlobalScorePct": "Score (%)", "JobTitle": "Job Title"},
        color="GlobalScorePct",
        color_continuous_scale="viridis",
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # 6) Visualization: Top 10 Competency Matches 
    st.markdown("### Competency Matches")
    try:
        # We encode the user's full text
        user_vec = model.encode(user_text, convert_to_tensor=True)
        comp_rows = []
        # We compute cosine similarity between user text and all competencies
        for comp_id, emb in comp_embs.items():
            sim = util.cos_sim(user_vec, emb).item()
            pct = round(100.0 * (sim + 1.0) / 2.0, 1)  # map [-1,1] -> [0,100]
            comp_rows.append((comp_id, sim, pct))

        comp_df = pd.DataFrame(comp_rows, columns=["CompetencyID", "CosSim", "MatchPct"])
        names_df = comp_unique[["CompetencyID", "CompetencyName"]]
        comp_df = comp_df.merge(names_df, on="CompetencyID", how="left")
        top10_comp = comp_df.sort_values("MatchPct", ascending=False).head(10)

        # We display the top 10 competencies as a horizontal bar chart
        fig_comp = px.bar(
            top10_comp.sort_values("MatchPct", ascending=True),
            x="MatchPct",
            y="CompetencyName",
            orientation="h",
            text="MatchPct",
            labels={"MatchPct": "Match (%)", "CompetencyName": "Competency"},
        )
        fig_comp.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
        fig_comp.update_layout(
            title="Top 10 Competency Matches (based on text)",
            xaxis=dict(range=[0, 100]),
            title_font_color="#e5e7eb",
            plot_bgcolor="#1c1c1c",
            paper_bgcolor="#1c1c1c",
            font_color="#e5e7eb",
            margin=dict(l=10, r=20, t=60, b=10),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute competency-level visualization: {e}")

