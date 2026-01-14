import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# =========================
# Configuration page
# =========================
st.set_page_config(
    page_title="Analyse des avis clients – Livres",
    layout="wide"
)

# =========================
# Chargement des données
# =========================
st.title("Analyse des avis clients – Livres")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "C:\\Users\\olake\\Desktop\\PROJETS_CAS_ENTREPRISE\\PROJET_1\\data\\donnees_nettoy\\avis_sentiments.csv"
data = pd.read_csv(DATA_PATH)
data['Review Date'] = pd.to_datetime(data['Review Date'], errors='coerce')

# =========================
# Labels des clusters
# =========================
cluster_labels = {
    0: "Auteurs et personnages",
    1: "Thèmes et contexte de l'histoire",
    2: "Appréciation de l'écriture et de l'auteur",
    3: "Réactions personnelles et critiques",
    4: "Expressions émotionnelles"
}

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Aller à",
    ["Vue par cluster","Analyse des sentiments","Avis exemples","Statistiques globales","Exporter PDF"]
)
st.sidebar.markdown("---")
dark_mode = st.sidebar.checkbox("Activer le mode sombre")

if dark_mode:
    st.markdown("""
    <style>
    .stApp {background-color: #0b1e3f; color: #e6e6e6;}
    h1, h2, h3, h4 {color: #fafafa;}
    .stDataFrame {background-color: #1c2a50; color: #e6e6e6;}
    div[data-testid="stSidebar"] {background-color: #091c35;}
    .stMetric {background-color: #12264e; padding: 15px; border-radius: 10px; color: #fafafa;}
    .stRadio > div {color: #e6e6e6;}
    </style>
    """, unsafe_allow_html=True)

# =========================
# Sélection du cluster
# =========================
clusters = sorted(data['cluster'].unique())
selected_cluster = st.sidebar.selectbox("Choisir un cluster", clusters)
cluster_data = data[data['cluster'] == selected_cluster]

# =========================
# Palette couleurs WordCloud
# =========================
palette_clusters = {
    0: ["#1f77b4", "#aec7e8"],
    1: ["#ff7f0e", "#ffbb78"],
    2: ["#2ca02c", "#98df8a"],
    3: ["#d62728", "#ff9896"],
    4: ["#9467bd", "#c5b0d5"]
}

def couleur_mots(word, font_size, position, orientation, random_state=None, **kwargs):
    couleurs = palette_clusters.get(selected_cluster, ["#444444", "#888888"])
    return random.choice(couleurs)

# =========================
# Fonction KPI
# =========================
def calcul_kpi(df):
    nb_avis = len(df)
    score_moyen = df['sentiment_score'].mean()
    labels = df['sentiment_label'].astype(str).str.lower()
    pct_pos = (labels == 'positive').mean() * 100
    pct_neg = (labels == 'negative').mean() * 100
    return nb_avis, score_moyen, pct_pos, pct_neg

# =========================
# Pages du menu
# =========================
if menu == "Vue par cluster":
    st.subheader(f"Cluster {selected_cluster} – {cluster_labels[selected_cluster]} ({len(cluster_data)} avis)")
    col1, col2 = st.columns([2,1])

    # --- WordCloud ---
    with col1:
        st.subheader("Nuage de mots")
        cluster_text = " ".join(cluster_data['avis_nettoy'].dropna())
        wordcloud = WordCloud(
            width=700, height=400,
            background_color='white',
            stopwords=STOPWORDS,
            max_words=100,
            color_func=couleur_mots
        ).generate(cluster_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # --- KPI ---
    with col2:
        st.subheader("Indicateurs clés")
        nb_avis, score_moyen, pct_pos, pct_neg = calcul_kpi(cluster_data)
        st.metric("Nombre d'avis", nb_avis)
        st.metric("Score moyen", f"{score_moyen:.2f}")
        st.metric("Avis positifs (%)", f"{pct_pos:.1f}%")
        st.metric("Avis négatifs (%)", f"{pct_neg:.1f}%")

elif menu == "Analyse des sentiments":
    st.subheader("Répartition des sentiments")
    sentiment_counts = cluster_data['sentiment_label'].value_counts()
    st.dataframe(sentiment_counts)
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["#2ca02c","#d62728","#7f7f7f"])
    ax.set_ylabel("Nombre d'avis")
    ax.set_xlabel("Sentiment")
    st.pyplot(fig)

elif menu == "Avis exemples":
    st.subheader("Avis représentatifs du cluster")
    avis_exemples = cluster_data['avis_nettoy'].dropna().head(6)
    cols = st.columns(3)
    for i, avis in enumerate(avis_exemples):
        with cols[i%3]:
            st.info(avis[:400]+"..." if len(avis)>400 else avis)

elif menu == "Statistiques globales":
    st.subheader("Statistiques globales par cluster")
    stats_globales = data.groupby('cluster')['sentiment_score'].agg(['count','mean'])
    st.dataframe(stats_globales)

elif menu == "Exporter PDF":
    st.subheader("Générer le PDF complet du dashboard")
    nb_avis, score_moyen, pct_pos, pct_neg = calcul_kpi(cluster_data)

    if st.button("📄 Générer PDF"):

        # -------------------
        # WordCloud
        cluster_text = " ".join(cluster_data['avis_nettoy'].dropna())
        wordcloud = WordCloud(width=700, height=400, background_color='white',
                              stopwords=STOPWORDS, max_words=100, color_func=couleur_mots).generate(cluster_text)
        buf_wc = BytesIO()
        wordcloud.to_image().save(buf_wc, format='PNG')
        buf_wc.seek(0)

        # -------------------
        # Histogramme sentiments
        fig_sent, ax_sent = plt.subplots()
        sentiment_counts = cluster_data['sentiment_label'].value_counts()
        ax_sent.bar(sentiment_counts.index, sentiment_counts.values, color=["#2ca02c","#d62728","#7f7f7f"])
        ax_sent.set_ylabel("Nombre d'avis")
        ax_sent.set_xlabel("Sentiment")
        buf_sent = BytesIO()
        fig_sent.savefig(buf_sent, format='PNG', bbox_inches='tight')
        buf_sent.seek(0)

        # -------------------
        # Évolution temporelle
        temp_data = cluster_data.dropna(subset=['Review Date'])
        sentiment_time = temp_data.set_index('Review Date').resample('M')['sentiment_score'].mean().reset_index()
        fig_time, ax_time = plt.subplots()
        ax_time.plot(sentiment_time['Review Date'], sentiment_time['sentiment_score'], marker='o')
        ax_time.set_title("Évolution du sentiment moyen")
        ax_time.set_xlabel("Temps")
        ax_time.set_ylabel("Score de sentiment")
        ax_time.grid(True)
        buf_time = BytesIO()
        fig_time.savefig(buf_time, format='PNG', bbox_inches='tight')
        buf_time.seek(0)

        # -------------------
        # Création PDF
        pdf_buf = BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height-50, "Dashboard Analyse des avis clients – Livres")
        c.setFont("Helvetica", 12)
        c.drawString(50, height-70, f"Cluster {selected_cluster} – {cluster_labels[selected_cluster]}")
        c.drawString(50, height-90, f"Date : {datetime.now().strftime('%d/%m/%Y')}")

        # KPIs
        c.drawString(50, height-120, f"Nombre d'avis : {nb_avis}")
        c.drawString(50, height-140, f"Score moyen : {score_moyen:.2f}")
        c.drawString(50, height-160, f"Avis positifs : {pct_pos:.1f}%")
        c.drawString(50, height-180, f"Avis négatifs : {pct_neg:.1f}%")

        # WordCloud
        c.drawImage(ImageReader(buf_wc), 50, height-500, width=500, height=300)
        # Histogramme sentiments
        c.drawImage(ImageReader(buf_sent), 50, height-830, width=500, height=250)
        # Graphique évolution
        c.showPage()
        c.drawImage(ImageReader(buf_time), 50, height-500, width=500, height=300)

        c.save()
        pdf_buf.seek(0)

        st.success("PDF généré !")
        st.download_button("⬇️ Télécharger le PDF", data=pdf_buf, file_name="dashboard_sentiments.pdf", mime="application/pdf")
