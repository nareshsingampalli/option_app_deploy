"""FileStorageBackend — saves results as CSV + JSON meta files."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from storage.base import SaveContext, StorageBackend, StorageHandler


def _filename_pair(ctx: SaveContext) -> tuple[str, str]:
    """Return (csv_path, meta_path) for the given context, using subdirectories."""
    # Map prefix to directory name
    dir_name = "nse_data" if ctx.prefix == "option" else "mcx_data"
    
    # Ensure directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    sym = ctx.symbol.lower()
    if ctx.time_str:
        clean = ctx.time_str.replace(":", "")
        csv_file  = f"{ctx.prefix}_{sym}_tabular_{ctx.date_str}_{clean}.csv"
        meta_file = f"{ctx.prefix}_{sym}_meta_{ctx.date_str}_{clean}.json"
    else:
        csv_file  = f"{ctx.prefix}_{sym}_tabular_{ctx.date_str}.csv"
        meta_file = f"{ctx.prefix}_{sym}_meta_{ctx.date_str}.json"
        
    return os.path.join(dir_name, csv_file), os.path.join(dir_name, meta_file)


class FileStorageBackend(StorageBackend):
    """Saves option data to CSV and metadata to JSON in the current directory."""

    def can_handle(self, ctx: SaveContext) -> bool:
        return True   # always available

    def save(self, ctx: SaveContext) -> bool:
        csv_path, meta_path = _filename_pair(ctx)

        # In live mode (no time suffix), delete stale files before writing
        if not ctx.time_str:
            for p in (csv_path, meta_path):
                if os.path.exists(p):
                    os.remove(p)

        meta = {
            "spot_price":        ctx.spot_price,
            "target_date":       ctx.date_str,
            "target_time":       ctx.time_str,
            "expiry_date":       ctx.expiry_dt.strftime("%Y-%m-%d") if ctx.expiry_dt else None,
            "fetched_at":        datetime.now().isoformat(),
            "has_data":          bool(ctx.rows),
            "expired_contracts": ctx.is_expired,
            "is_fallback":       ctx.is_fallback,
        }

        if ctx.rows:
            pd.DataFrame(ctx.rows).to_csv(csv_path, index=False)
            print(f"[FileStorage] Saved {len(ctx.rows)} rows -> {csv_path}")
        else:
            print(f"[FileStorage] No rows to save.")

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[FileStorage] Meta -> {meta_path}")
        return True


# ── Chain handler wrapping FileStorageBackend ────────────────────────────────

class FileStorageHandler(StorageHandler):
    """Chain node that delegates to FileStorageBackend."""

    def __init__(self):
        super().__init__()
        self._backend = FileStorageBackend()

    def handle(self, ctx: SaveContext) -> bool:
        if self._backend.can_handle(ctx):
            return self._backend.save(ctx)
        return self._forward(ctx)
