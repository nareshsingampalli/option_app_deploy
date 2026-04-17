
import requests

url = "http://localhost:8010/api/spot-probe?exchange=NSE&symbol=NIFTY"
try:
    resp = requests.get(url)
    print(f"Status: {resp.status_code}")
    print(f"JSON: {resp.json()}")
except Exception as e:
    print(f"Error: {e}")
