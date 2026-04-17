
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from resolvers.mcx_resolver import MCXInstrumentResolver
from core.utils import ist_now

def test_resolver():
    today = ist_now().strftime("%Y-%m-%d")
    print(f"Testing MCX resolver for CRUDEOIL on {today}")
    resolver = MCXInstrumentResolver()
    key = resolver.get_spot_key("CRUDEOIL", today)
    print(f"Resolved key: {key}")

if __name__ == "__main__":
    test_resolver()
