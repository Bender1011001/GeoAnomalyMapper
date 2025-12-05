import requests
from bs4 import BeautifulSoup

url = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/87/087A_05304_121415/epochs/"
print(f"Scanning: {url}")

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    
    print("Epochs content:")
    for link in links:
        print(link.get('href'))
            
except Exception as e:
    print(f"Error: {e}")