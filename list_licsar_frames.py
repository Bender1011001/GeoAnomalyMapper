import requests
from bs4 import BeautifulSoup
import sys

base_url = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/87/"
print(f"Scanning: {base_url}")

try:
    response = requests.get(base_url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    
    print("Available Frames:")
    for link in links:
        href = link.get('href').strip('/')
        if href.startswith('087'):
            print(href)
            
except Exception as e:
    print(f"Error: {e}")