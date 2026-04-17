
import upstox_client

print(f"upstox_client classes: {dir(upstox_client)}")
try:
    api = upstox_client.HistoryV3Api()
    print("HistoryV3Api exists")
except AttributeError:
    print("HistoryV3Api does NOT exist")
except Exception as e:
    print(f"Error accessing HistoryV3Api: {e}")

try:
    api = upstox_client.HistoryApi()
    print("HistoryApi exists")
except AttributeError:
    print("HistoryApi does NOT exist")

