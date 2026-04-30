import requests
from core.config import UPSTOX_ACCESS_TOKEN, UPSTOX_API_URL

def test_token():
    url = f"{UPSTOX_API_URL}/v2/user/profile"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
    }
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_token()
