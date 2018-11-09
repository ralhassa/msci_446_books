import wikipedia
import pandas as pd
import csv
from bs4 import BeautifulSoup

# main page https://en.wikipedia.org/wiki/Lists_of_fiction_works_made_into_feature_films
# page = wikipedia.WikipediaPage("List_of_fiction_works_made_into_feature_films_(0–9,_A–C)")
# page = wikipedia.WikipediaPage("List_of_fiction_works_made_into_feature_films_(D–J)")
# page = wikipedia.WikipediaPage("List_of_fiction_works_made_into_feature_films_(K-R)")

# reads in wikipedia page
page = wikipedia.WikipediaPage("List_of_fiction_works_made_into_feature_films_(S-Z)")
# gets html of the page
html = page.html()

soup = BeautifulSoup(html, 'html.parser')
# finds all relevant tables in the page
tables = soup.find_all('table', attrs={'class':'wikitable'})  # finds tables, puts into array

book_name = list()
film_name = list()

# gathers the book names and movie names in the table and puts them into a list
for table in tables:
    rows = table.find_all('tr')
    for tr in rows:
        cols = tr.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if len(cols) == 2:
            book_name.append(cols[0])
            film_name.append(cols[1])

# writes to csv, unformatted
with open('output.csv', 'w', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(book_name)
    writer.writerow(film_name)

csv_file.close()
