import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://books.toscrape.com/catalogue/page-{}.html"
NUM_PAGES = 5  # nombre de pages à scraper

books = []

for page in range(1, NUM_PAGES + 1):
    url = BASE_URL.format(page)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Erreur page {page}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    products = soup.find_all("article", class_="product_pod")

    for p in products:
        title = p.h3.a["title"]
        price = p.find("p", class_="price_color").text
        rating_class = p.p["class"][1]  # Ex: "Three"
        rating_dict = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}
        rating = rating_dict.get(rating_class, 0)

        books.append({
            "title": title,
            "price": price,
            "rating": rating
        })

    time.sleep(1)  # respect site

df = pd.DataFrame(books)
df.to_csv("data/donnees_brutes/books.csv", index=False)
print(f"{len(df)} livres collectés")
