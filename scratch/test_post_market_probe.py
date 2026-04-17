
import requests
import json

url = "http://localhost:8010/api/pre-market-status?exchange=NSE"
try:
    resp = requests.get(url)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
