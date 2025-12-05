import requests

base_url = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/"
print(f"Checking base URL: {base_url}")
r = requests.get(base_url)
print(f"Base URL Status: {r.status_code}")

track_url = base_url + "87/"
print(f"Checking track URL: {track_url}")
r = requests.get(track_url)
print(f"Track URL Status: {r.status_code}")

# Try with leading zero
track_url_0 = base_url + "087/"
print(f"Checking track URL (087): {track_url_0}")
r = requests.get(track_url_0)
print(f"Track URL (087) Status: {r.status_code}")