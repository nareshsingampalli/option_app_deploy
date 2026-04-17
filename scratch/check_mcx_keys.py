
import gzip
import json
import os

def find_crudeoil():
    path = "scratch/MCX.json.gz"
    if not os.path.exists(path):
        print("File not found.")
        return
    
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total MCX instruments: {len(data)}")
    
    crude_futs = [inst for inst in data if "CRUDEOIL" in inst['trading_symbol'] and "FUT" in inst['trading_symbol']]
    print(f"Found {len(crude_futs)} CRUDEOIL FUT instruments.")
    for fut in crude_futs[:5]:
        print(f"Symbol: {fut['trading_symbol']}, Key: {fut['instrument_key']}, Expiry: {fut['expiry']}")

if __name__ == "__main__":
    find_crudeoil()
