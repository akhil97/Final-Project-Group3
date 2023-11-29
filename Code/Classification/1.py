import requests
from bs4 import BeautifulSoup
import time
import json
import os


def scrape_supreme_court_cases(base_url):
    links = []
    no_of_pages = list(range(100))  # Adjust the number of pages as needed

    # Iterate through pages
    for page_in in no_of_pages:
        time.sleep(1)
        URL = f"{base_url}&pagenum={page_in}"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        result_new3 = soup.find_all(class_='result_url')

        if not result_new3:
            break  # No more pages

        for link_new3 in result_new3:
            # Extracting case details here, modify based on actual HTML structure
            case_title = link_new3.find('a').text
            case_url = base_url + link_new3['href']
            links.append({'title': case_title, 'url': case_url})

    return links


def scrape_supreme_court(base_url):
    all_links = {}

    # Iterate through years
    for year in range(1947, 2021):
        time.sleep(1)
        URL = f"{base_url}{year}/"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        result_new2 = soup.find_all(class_='browselist')

        year_links = []
        for link_new2 in result_new2:
            print(f"{year}: {link_new2.find('a').text} Year Started ...")
            year_links += scrape_supreme_court_cases(base_url + link_new2.find('a')['href'])

        all_links[year] = year_links

    return all_links


def main():
    base_url = 'https://indiankanoon.org/browse/supremecourt/'
    supreme_court_links = scrape_supreme_court(base_url)

    # Create a directory to store files based on the year
    output_directory = '/home/ubuntu/NLP_Project/Code/Classification/Sentiment/cases_by_year'
    os.makedirs(output_directory, exist_ok=True)

    for year, links in supreme_court_links.items():
        output_file = os.path.join(output_directory, f'court_cases_{year}.json')
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(links, outfile, indent=4)


if __name__ == "__main__":
    main()
