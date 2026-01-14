# Importation des packages nécessaires au projet
import requests # requests sert à envoyer des requêtes HTTP à un site web
from bs4 import BeautifulSoup # ce package permet d'analyser le code HTML d'une page web
import pandas as pd # stocker les données dans un dataframe
import time # pour faire des pauses entre les requêtes


# Site utilisé pour le scraping :https://books.toscrape.com/
# L'url du site à scraper
# J'ai choisi un site de scraping légal. Il s'agit d'un site publiant des livres où les clients
# donnent leurs avis, des notes étoilées allant de 1 à 5.
url_site = "http://books.toscrape.com/catalogue/page-{}.html"

# J'ai choisi de scraper 50 pages.
nbre_pages = 50  # nombre de pages à scraper

# Je définis ici la liste où seront stockés les livres récupérés
livres = []

# Je scrape page par page :
for page in range(1, nbre_pages + 1):

    # le numéro de page est remplacé dans l'url, une requête est envoyée au site, 
    # le contenu HTML de la page est récupéré, si la page est bien chargée, la page
    # suivante est scrapée et ainsi de suite.
    url = url_site.format(page)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Erreur page {page}")
        continue

    # Le HTML brut est transformé en un objet analysable, facilitant la recherche des balises
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Tous les blocs correspondant à un livre sont sélectionnés. Après l'analyse du code HTML, on 
    # constate que chaque livre est dans une balise <article class ="product_prod">
    blocs_livres = soup.find_all("article", class_="product_pod")

    # Je récupère des informations sur chaque livre trouvé sur la page
    for p in blocs_livres:

        # Le titre du livre est récupéré
        titre = p.h3.a["title"]

        # Son prix également
        prix = p.find("p", class_="price_color").text

        # ainsi que la note qui lui a été attribuée
        note_categ = p.p["class"][1] 

        # C'est le dictionnaire où les notes, de base en lettres (One, Two, ...) sont transformés en chiffres 
        # pour faciliter l'analyse
        dict_notes = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}

        # les notes textes sont converties en notes numériques. En cas de souci,
        # la note est égale à 0 par défaut
        note = dict_notes.get(note_categ, 0)

        # on ajoute le livre récupéré avec ses infos à la liste des livres
        livres.append({
            "titre": titre,
            "prix": prix,
            "note": note
        })

    # Il y a une pause d'une seconde avant la page suivante pour un scraping responsable et légal
    time.sleep(1)  

# La liste de livres est transformée en dataframe
df = pd.DataFrame(livres)

# Et ensuite sauvegardé en fichier CSV
df.to_csv("data/donnees_brutes/livres.csv", index=False)
print(f"{len(df)} livres collectés")
