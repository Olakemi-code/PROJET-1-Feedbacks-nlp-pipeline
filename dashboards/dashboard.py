import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from fpdf import FPDF
from collections import Counter
import spacy

# =========================
# Charger SpaCy et stopwords anglais
# =========================
nlp = spacy.load("en_core_web_sm")  # modèle anglais
english_stopwords = list(nlp.Defaults.stop_words)

# =========================
# Chargement des données
# =========================
st.title("Customer Feedback Analysis – Book Reviews")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "donnees_nettoy" / "reviews_clean.csv"
data = pd.read_csv(DATA_PATH)

# =========================
# Nettoyage et lemmatisation
# =========================
def spacy_clean(text):
    doc = nlp(str(text).lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

data["clean_text"] = data["title"].apply(spacy_clean)

# =========================
# Paramètres clustering
# =========================
st.subheader("Paramètres - Clustering")
clustering_method = st.selectbox("Choose clustering method", ["KMeans", "LDA"])
k = st.slider("Number of clusters/topics", min_value=2, max_value=10, value=5)

vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words=english_stopwords)
X_tfidf = vectorizer.fit_transform(data["clean_text"])

if clustering_method == "KMeans":
    model = KMeans(n_clusters=k, random_state=42)
    data["cluster"] = model.fit_predict(X_tfidf)
else:
    model = LatentDirichletAllocation(n_components=k, random_state=42)
    topic_distrib = model.fit_transform(X_tfidf)
    data["cluster"] = topic_distrib.argmax(axis=1)

# =========================
# Mots clés et labels
# =========================
def get_top_terms(model, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    if clustering_method == "KMeans":
        centers = model.cluster_centers_
    else:  # LDA
        centers = model.components_
    for i in range(centers.shape[0]):
        top_idx = centers[i].argsort()[-n_terms:][::-1]
        cluster_keywords[i] = [terms[j] for j in top_idx]
    return cluster_keywords

cluster_keywords = get_top_terms(model, vectorizer)
cluster_labels = {i: " / ".join(words[:3]) for i, words in cluster_keywords.items()}

# =========================
# Couleurs clusters
# =========================
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
def color_func(cluster_id):
    def random_color(word=None, font_size=None, position=None, orientation=None, **kwargs):
        return cluster_colors[cluster_id % len(cluster_colors)]
    return random_color

# =========================
# Sélection cluster et WordCloud
# =========================
st.subheader("Analyse des thèmes")
clusters = sorted(data["cluster"].unique())
selected_cluster = st.selectbox("Select a cluster", clusters)
st.markdown(f"**Cluster {selected_cluster} : {cluster_labels[selected_cluster]}**")

cluster_texts = data[data["cluster"] == selected_cluster]["clean_text"]
cluster_text = " ".join(cluster_texts)

wordcloud = WordCloud(
    width=600, height=400, background_color="white", max_words=100,
    color_func=color_func(selected_cluster), stopwords=STOPWORDS
).generate(cluster_text)

word_counts = Counter(" ".join(cluster_texts).split())
top_words = dict(word_counts.most_common(10))

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(wordcloud, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("WordCloud")
axes[1].barh(list(top_words.keys())[::-1], list(top_words.values())[::-1],
             color=cluster_colors[selected_cluster % len(cluster_colors)])
axes[1].set_title("Top 10 words")
axes[1].set_xlabel("Frequency")
plt.tight_layout()
st.pyplot(fig)

# =========================
# Exemples d'avis
# =========================
st.subheader("Exemples d'avis dans ce cluster")
for e in cluster_texts.head(5).tolist():
    st.write(f"- {e}")

# =========================
# Sentiment par thème
# =========================
st.subheader("Sentiment par thème")
sentiment_theme = data.groupby(["cluster", "sentiment"]).size().unstack(fill_value=0)
st.dataframe(sentiment_theme)
st.bar_chart(sentiment_theme)

# =========================
# Filtre par sentiment
# =========================
st.subheader("Filtre par Sentiment")
selected_sentiment = st.radio("Select sentiment", data["sentiment"].unique())
filtered_data = data[data["sentiment"] == selected_sentiment]
st.dataframe(filtered_data[["title", "numeric_rating", "sentiment", "cluster"]])

# =========================
# Export CSV / PDF
# =========================
st.subheader("Export CSV / PDF")
enable_pdf = st.checkbox("Enable PDF export", value=True)

st.download_button(
    label="Download CSV",
    data=filtered_data.to_csv(index=False),
    file_name=f"reviews_{selected_sentiment}.csv",
    mime="text/csv"
)

if enable_pdf:
    def create_pdf(df, labels, filename="feedback_report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Customer Feedback Analysis", ln=True, align='C')
        pdf.ln(5)
        for cluster_id, label in labels.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"Cluster {cluster_id} : {label}", ln=True)
            pdf.set_font("Arial", '', 11)
            examples = df[df["cluster"]==cluster_id]["clean_text"].head(3).tolist()
            for e in examples:
                pdf.multi_cell(0,8,f"- {e}")
            pdf.ln(3)
        pdf.output(filename)
        return filename
    pdf_file = create_pdf(filtered_data, cluster_labels)
    st.download_button(
        label="Download PDF",
        data=open(pdf_file,"rb").read(),
        file_name="feedback_report.pdf",
        mime="application/pdf"
    )
