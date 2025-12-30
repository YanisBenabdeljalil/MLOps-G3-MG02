# 🚀 Match My Skills - Cloud Edition

**Match My Skills** est un moteur de recommandation de métiers intelligent basé sur le **NLP (Natural Language Processing)**. Il permet aux utilisateurs de découvrir les intitulés de poste les plus adaptés à leur profil grâce à une analyse sémantique profonde.

L'application utilise des modèles de **Deep Learning (Sentence Transformers)** pour comparer le profil d'un candidat avec un référentiel de compétences métier, calculant un score de compatibilité précis.

---

## 📂 Structure du Projet

L'organisation du dépôt suit les meilleures pratiques MLOps pour séparer l'infrastructure, le traitement des données et l'application.

### 🔹 `src/` : Cœur de l'application
* **`api/main.py`** : Serveur **FastAPI** exposant les endpoints de prédiction, de santé (`/health`) et de métriques.
* **`data/`** : Pipeline de données complet.
    * `competency_job.csv` : **Dataset de référence** contenant les compétences et intitulés de postes.
    * `clean_transform.py` : Logique de nettoyage et de préparation des données.
    * `data_pipeline.py` : Orchestrateur du flux de données.
    * `download_data.py` & `load_final.py` : Gestion des transferts avec **AWS S3**.
* **`model/`** : Scripts d'entraînement (`train.py`) et de vérification (`verify.py`) du modèle.

### 🔹 `terraform/` : Infrastructure as Code (IaC)
* **`main.tf`** : Définition de l'infrastructure Cloud sur **AWS** (S3, ECR, et App Runner).

### 🔹 `.github/workflows/` : Pipeline CI/CD
* **`cicd.yml`** : Automatisation complète. À chaque push, GitHub exécute les tests, construit l'image Docker et la déploie sur **AWS ECR**.
* **`test-aws.yml`** : Workflow de vérification de la connectivité avec les services AWS.

### 🔹 `tests/` : Assurance Qualité
* **`test_api.py`** : Tests d'intégration pour valider le bon fonctionnement des endpoints de l'API.
* **`test_data.py`** : Tests unitaires garantissant la fiabilité des transformations de données.

### 🔹 Racine du projet
* **`app.py`** : Interface utilisateur développée avec **Streamlit**.
* **`Dockerfile`** : Instructions de conteneurisation pour le **Backend** (FastAPI).
* **`Dockerfile.frontend`** : Instructions de conteneurisation dédiées au **Frontend** (Streamlit).
* **`requirements.txt`** : Liste des dépendances Python indispensables au projet (FastAPI, Streamlit, PyTorch, etc.).

---

## 🏗️ Architecture Technique
L'architecture est de type **Microservices découplée** pour garantir la scalabilité:
1. **Backend** : FastAPI hébergé sur **AWS App Runner**.
2. **Frontend** : Interface Streamlit (utilisable en mode Cloud via AWS App Runner ou localement en mode "Hybride").
3. **Stockage** : **AWS S3** pour les modèles et datasets, **AWS ECR** pour le registre d'images Docker.

---

## 🚀 Installation Rapide pour lancer le Frontend en local
(Si le frontend Streamlit créé sur AWS App Runner ne charge pas) : 

1. **Installation des dépendances :**
   ```bash
   pip install -r requirements.txt

2. **Lancement local du Frontend :**
   ```bash
   streamlit run app.py