"""
DataIngester + UserSequenceGenerator

DataIngester: reads events_clean.parquet and bulk-inserts into MongoDB
real collections (one collection per event type).

UserSequenceGenerator: reads real_* collections and writes per-user
interaction sequences to real_sequences. Used by the downstream RecSys
evaluator (GRU4Rec needs session-ordered sequences per user).

MongoDB collections (real):
    real_page_visit, real_search_query, real_add_to_cart,
    real_remove_from_cart, real_product_buy

MongoDB collection (sequences):
    real_sequences

Document schema (event collections):
    {client_id: int, timestamp: ISODate, event_type: str,
     sku: int|null, session_id: int|null}

Indexes: compound {client_id: 1, timestamp: 1} on each event collection.

See: mermaid/Level4_DataIngestionUpload.md (SyneriseDataIngestor,
     UserSequenceGenerator)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

import pandas as pd
import pymongo
from pymongo import MongoClient

from config import CLEAN_PARQUET, MONGO_DB, MONGO_URI, SESSION_TIMEOUT_MIN

REAL_COLLECTIONS = {
    "page_visit":       "real_page_visit",
    "search_query":     "real_search_query",
    "add_to_cart":      "real_add_to_cart",
    "remove_from_cart": "real_remove_from_cart",
    "product_buy":      "real_product_buy",
}


class DataIngester:
    """
    Reads events_clean.parquet and bulk-inserts into MongoDB real collections.
    One collection per event type for fast per-type queries.

    Idempotent: drops and recreates real_* collections on each call to ingest().
    Processes in chunks of CHUNK_SIZE rows to stay within memory bounds.
    """

    CHUNK_SIZE = 50_000

    def __init__(self):
        self.mongo_uri = MONGO_URI
        self.db_name   = MONGO_DB

    def ingest(self, parquet_path=None) -> None:
        """
        Full pipeline: read parquet -> split by event_type -> chunk-insert each.
        Prints progress per event type.
        """
        path = parquet_path or CLEAN_PARQUET
        print(f"\nDataIngester: reading {path} ...")

        df = pd.read_parquet(
            path,
            columns=["client_id", "timestamp", "event_type", "session_id", "sku"],
        )
        print(f"  Loaded {len(df):,} events.")

        client = MongoClient(self.mongo_uri)
        db     = client[self.db_name]

        # Drop existing real_* collections (idempotent re-ingest)
        for coll_name in REAL_COLLECTIONS.values():
            db.drop_collection(coll_name)
            print(f"  Dropped {coll_name}")

        # Insert per event type
        for etype, coll_name in REAL_COLLECTIONS.items():
            subset = df[df["event_type"] == etype]
            n      = len(subset)
            if n == 0:
                print(f"  {coll_name}: no rows, skipping")
                continue

            coll   = db[coll_name]
            n_chunks = math.ceil(n / self.CHUNK_SIZE)
            inserted = 0

            for i in range(n_chunks):
                chunk = subset.iloc[i * self.CHUNK_SIZE : (i + 1) * self.CHUNK_SIZE]
                records = self._rows_to_docs(chunk)
                self._insert_chunk(coll, records)
                inserted += len(records)
                print(
                    f"  {coll_name}: {inserted:,}/{n:,} ({100*inserted//n}%)",
                    end="\r",
                )

            print(f"  {coll_name}: {inserted:,} docs inserted.           ")

        self._ensure_indexes(db)
        client.close()
        print("\nDataIngester: done.")

    def _rows_to_docs(self, df: pd.DataFrame) -> List[dict]:
        """Convert a DataFrame chunk to a list of BSON-safe dicts."""
        docs = []
        for row in df.itertuples(index=False):
            sku = row.sku
            docs.append({
                "client_id":  int(row.client_id),
                "timestamp":  _to_datetime(row.timestamp),
                "event_type": row.event_type,
                "sku":        int(sku) if pd.notna(sku) else None,
                "session_id": int(row.session_id) if pd.notna(row.session_id) else None,
            })
        return docs

    def _insert_chunk(self, collection, records: List[dict]) -> None:
        """Bulk insert; ordered=False allows partial success on duplicate errors."""
        if records:
            collection.insert_many(records, ordered=False)

    def _ensure_indexes(self, db) -> None:
        """Compound index on (client_id, timestamp) for each real_* collection."""
        for coll_name in REAL_COLLECTIONS.values():
            db[coll_name].create_index(
                [("client_id", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)],
                background=True,
            )
        print("  Indexes created on all real_* collections.")


class UserSequenceGenerator:
    """
    Reads real_* MongoDB collections and writes per-user interaction sequences
    to real_sequences. Each document stores the last window_size sessions for
    one user, suitable as input to sequence-based RecSys models (GRU4Rec).

    Document schema (real_sequences):
        {
          client_id : int,
          sequences : [
            [{event_type, sku, timestamp}, ...],  // session 0 (oldest)
            [{event_type, sku, timestamp}, ...],  // session 1
            ...
          ]
        }
    """

    USER_BATCH = 1_000    # users processed per MongoDB write batch

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.mongo_uri   = MONGO_URI
        self.db_name     = MONGO_DB

    def generate_sequences(self, client_ids: Optional[List[int]] = None) -> None:
        """
        Generate and write sequences for all (or given) client_ids.

        Reads from real_* collections, sessionizes at SESSION_TIMEOUT_MIN gap,
        keeps the last window_size sessions per user, writes to real_sequences.
        client_ids=None -> process all users found in real_* collections.
        """
        client = MongoClient(self.mongo_uri)
        db     = client[self.db_name]

        seq_coll = db["real_sequences"]
        seq_coll.drop()

        # Collect all events for the requested users from all real_* collections
        print("\nUserSequenceGenerator: fetching events ...")
        pipeline = []
        if client_ids:
            pipeline.append({"$match": {"client_id": {"$in": list(client_ids)}}})
        pipeline += [
            {"$project": {"_id": 0, "client_id": 1, "timestamp": 1,
                          "event_type": 1, "sku": 1}},
            {"$sort": {"client_id": 1, "timestamp": 1}},
        ]

        # Pull from all real_* collections
        all_events: List[dict] = []
        for coll_name in REAL_COLLECTIONS.values():
            all_events.extend(list(db[coll_name].aggregate(pipeline)))

        if not all_events:
            print("  No events found.")
            client.close()
            return

        print(f"  {len(all_events):,} events loaded.")

        # Group by client_id
        events_by_user: dict = {}
        for ev in all_events:
            cid = ev["client_id"]
            events_by_user.setdefault(cid, []).append(ev)

        timeout = pd.Timedelta(minutes=SESSION_TIMEOUT_MIN)
        docs    = []
        n_users = len(events_by_user)
        written = 0

        for cid, evs in events_by_user.items():
            evs.sort(key=lambda e: e["timestamp"])
            sessions = _sessionize_events(evs, timeout)

            # Keep last window_size sessions
            sessions = sessions[-self.window_size :]

            docs.append({"client_id": cid, "sequences": sessions})

            if len(docs) >= self.USER_BATCH:
                seq_coll.insert_many(docs, ordered=False)
                written += len(docs)
                print(f"  {written:,}/{n_users:,} users written", end="\r")
                docs = []

        if docs:
            seq_coll.insert_many(docs, ordered=False)
            written += len(docs)

        seq_coll.create_index("client_id", background=True)
        print(f"\n  {written:,} user sequence documents written.")
        client.close()


# Helpers

def _to_datetime(ts) -> datetime:
    """Convert pandas Timestamp or numpy datetime64 to tz-naive datetime."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t.to_pydatetime()


def _sessionize_events(
    events: List[dict],
    timeout: pd.Timedelta,
) -> List[List[dict]]:
    """
    Split a sorted event list into sessions (30-min inactivity gap).
    Returns list of sessions, each session a list of event dicts.
    """
    if not events:
        return []

    sessions, current = [], []
    prev_ts = None

    for ev in events:
        ts = pd.Timestamp(ev["timestamp"])
        if prev_ts is not None and (ts - prev_ts) > timeout:
            sessions.append(current)
            current = []
        current.append({
            "event_type": ev["event_type"],
            "sku":        ev.get("sku"),
            "timestamp":  ev["timestamp"],
        })
        prev_ts = ts

    if current:
        sessions.append(current)

    return sessions
