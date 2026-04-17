
import upstox_client
import inspect

print("HistoryApi.get_intra_day_candle_data signature:")
print(inspect.signature(upstox_client.HistoryApi.get_intra_day_candle_data))
print("\nHistoryV3Api.get_intra_day_candle_data signature:")
print(inspect.signature(upstox_client.HistoryV3Api.get_intra_day_candle_data))

print("\nHistoryApi.get_historical_candle_data1 signature:")
try:
    print(inspect.signature(upstox_client.HistoryApi.get_historical_candle_data1))
except AttributeError:
    print("get_historical_candle_data1 does NOT exist in HistoryApi")

print("\nHistoryV3Api.get_historical_candle_data1 signature:")
try:
    print(inspect.signature(upstox_client.HistoryV3Api.get_historical_candle_data1))
except AttributeError:
    print("get_historical_candle_data1 does NOT exist in HistoryV3Api")
