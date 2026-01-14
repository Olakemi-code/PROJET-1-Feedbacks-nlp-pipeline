# Automatisation intelligente de l’analyse des feedbacks clients

## Contexte
Les entreprises reçoivent quotidiennement, enormément de retours clients, c'est-à-dire des avis en ligne ou des formulaires ou des commentaires, des posts sur les réseaux sociaux. Ces données sont généralement non structurées, peu exploitées pour la prise de décision alors qu'elles contiennent des informations essentielles pour améliorer les produits, services proposés et donc, améliorer l'expérience client. 

## Objectif
L'objectif est donc de construire une chaîne automatisée permettant :
- collecter des feedbacks clients
- analyser leur contenu textuel
- identifier les thèmes récurrents
- restituer les insights métiers via un dashboard interactif

## Pipeline de traitement
1. Collecte des données
   Scraping légal d'avis clients depuis un site de livres.
2. Nettoyage et préparation NLP
- normalisation du texte
- suppression du bruit
- préparation des données pour le machine learning
3. Analyse thématique  
- vectorisation TF-IDF
- clustering non supervisé (KMeans/LDA)
- extraction des mots clés par cluster
- interpretation des clusters
4. Modélisation
- chargement d'un modèle pré-entrainé distilbert
- prédiction du sentiment (positif ou négatif) par cluster
4. Restitution  
- dashboard Streamlit
- visualisation des thèmes
- exploration des avis par cluster
  
## Technologies utilisées
- Python (scikit-learn, pandas, numpy)
- Web scraping (BeautifulSoup, Selenium)
- NLP(text mining)
- Streamlit (dashboard)
- Git/Github

## Structure du projet
PROJET_1/
├── data/
│   ├── donnees_brutes/
│   ├── donnees_nettoy/
│   └── sourc_ext/
├── src/
│   ├── scraping/
│   ├── preprocessing/
│   └── modeling/
├── dashboards/
│   └── dashboard.py
├── notebooks/
├── requirements.txt
└── README.md


## Dashboard
Le dashboard permet de :
- visualiser les indicateurs clés
- explorer les thèmes identifiés
- visualiser les sentiments par cluster
- consulter des exemples d'avis par cluster
