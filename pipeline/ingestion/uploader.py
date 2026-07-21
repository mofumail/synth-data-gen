"""
UploaderService

Background service: polls SYNTH_DIR every 10s for new Parquet batch files,
bulk-inserts them into MongoDB synthetic_* collections, then moves each file
to /archive.

Runs as a separate process alongside SimulationOrchestrator (step 5 in the
SLC1 sequence diagram).

MongoDB collections (synthetic):
    synthetic_page_visit, synthetic_search_query, synthetic_add_to_cart,
    synthetic_remove_from_cart, synthetic_product_buy

See: mermaid/new/PROPOSED_SLC1.md (step 5: ASYNC UPLOAD)
     mermaid/Level4_DataIngestionUpload.md (SyneriseUpdateUploader)
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from pymongo import MongoClient

from config import MONGO_DB, MONGO_URI, SYNTH_DIR

ARCHIVE_DIR     = SYNTH_DIR / "archive"
POLL_INTERVAL_S = 10

SYNTH_COLLECTIONS = {
    "page_visit":       "synthetic_page_visit",
    "search_query":     "synthetic_search_query",
    "add_to_cart":      "synthetic_add_to_cart",
    "remove_from_cart": "synthetic_remove_from_cart",
    "product_buy":      "synthetic_product_buy",
}


class UploaderService:
    """
    Background service: polls SYNTH_DIR every POLL_INTERVAL_S seconds for new
    *.parquet files, bulk-inserts them into synthetic_* collections, then
    moves each processed file to ARCHIVE_DIR.

    Usage (in a subprocess):
        from ingestion.uploader import UploaderService
        UploaderService().run()
    """

    def __init__(self):
        self.mongo_uri = MONGO_URI
        self.db_name   = MONGO_DB
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        Blocking poll loop. Run in a subprocess or background thread.
        Logs every scan cycle; exits cleanly on KeyboardInterrupt.
        """
        print(f"\nUploaderService: watching {SYNTH_DIR} (every {POLL_INTERVAL_S}s)")
        try:
            while True:
                n = self.scan_and_upload(SYNTH_DIR)
                if n:
                    print(f"  [{_now()}] uploaded {n} file(s).")
                time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            print("\nUploaderService: stopped.")

    def scan_and_upload(self, folder: Path, is_synthetic: bool = True) -> int:
        """
        Scan folder for *.parquet files, upload each, archive after.

        Args:
            folder: directory to scan
            is_synthetic: if True use synthetic_* collections; future-proofing
                hook for uploading real batches via the same service

        Returns:
            Number of files processed.
        """
        parquet_files = sorted(folder.glob("*.parquet"))
        if not parquet_files:
            return 0

        for path in parquet_files:
            try:
                self.upload_file(path)
            except Exception as e:
                print(f"  [WARN] Failed to upload {path.name}: {e}")

        return len(parquet_files)

    def upload_file(self, path: Path) -> None:
        """
        Read one Parquet batch, split by event_type, bulk-insert to
        synthetic_* collections, then move to ARCHIVE_DIR.
        """
        df = pd.read_parquet(path)
        n  = len(df)

        client = MongoClient(self.mongo_uri)
        db     = client[self.db_name]

        for etype, coll_name in SYNTH_COLLECTIONS.items():
            subset = df[df["event_type"] == etype]
            if subset.empty:
                continue
            records = self._rows_to_docs(subset)
            self._insert_chunk(db[coll_name], records)

        client.close()

        dest = ARCHIVE_DIR / path.name
        shutil.move(str(path), str(dest))
        print(f"  Uploaded {path.name} ({n:,} events) -> archived.")

    def _rows_to_docs(self, df: pd.DataFrame) -> List[dict]:
        """Convert DataFrame rows to BSON-safe dicts."""
        docs = []
        for row in df.itertuples(index=False):
            sku = getattr(row, "sku", None)
            sid = getattr(row, "session_id", None)
            docs.append({
                "client_id":  int(row.client_id),
                "timestamp":  _to_datetime(row.timestamp),
                "event_type": row.event_type,
                "sku":        int(sku) if sku is not None and pd.notna(sku) else None,
                "session_id": int(sid) if sid is not None and pd.notna(sid) else None,
            })
        return docs

    def _insert_chunk(self, collection, records: List[dict]) -> None:
        """insert_many with ordered=False for speed."""
        if records:
            collection.insert_many(records, ordered=False)


# Helpers

def _to_datetime(ts) -> datetime:
    """Convert pandas Timestamp to tz-naive Python datetime for BSON."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t.to_pydatetime()


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")
