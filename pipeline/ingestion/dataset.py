"""
SessionDataset + InteractionGenerator

PyTorch Dataset wrapping events_clean.parquet for SessionTransformer training.

Each item is one session: (events, items, deltas) tensors used as input to
SessionTransformer.forward(). The forward() method internally shifts by 1
(input t=0..T-2, target t=1..T-1), so we store the full session here.

Splits (from config):
    train : timestamp < TRAIN_CUTOFF
    val   : TRAIN_CUTOFF <= t < VAL_CUTOFF
    test  : timestamp >= VAL_CUTOFF
"""

from __future__ import annotations

from typing import Dict, List, Optional
import os

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict

from config import (
    CLEAN_PARQUET, HISTORY_WINDOW, OUTPUT_DIR, TRAIN_CUTOFF, VAL_CUTOFF, VOCAB_K,
    CAT2IDX_PATH, CAT_SKU_POOLS_PATH, SVDPQ_PATH,
)
from simulation.generator.session_transformer import (
    ACTION2IDX, BIN_EDGES, EOS_IDX, N_TEMPORAL_BINS, TEMPORAL_MIN_S, TEMPORAL_MAX_S,
)


def _cache_path(split: str, max_length: int, min_length: int, max_sessions) -> str:
    tag = f"{split}_ml{max_length}_min{min_length}_ms{max_sessions or 'all'}"
    return str(OUTPUT_DIR / f"session_cache_{tag}.joblib")


def _cache_valid(cache_file: str, parquet_path: str) -> bool:
    """Cache is valid if it exists and is newer than the source parquet."""
    if not os.path.exists(cache_file):
        return False
    return os.path.getmtime(cache_file) >= os.path.getmtime(parquet_path)

SKU2IDX_PATH        = OUTPUT_DIR / "sku2idx.joblib"
VOCAB_STATS_PATH    = OUTPUT_DIR / "vocab_stats.joblib"
SKU_PROPERTIES_PATH = OUTPUT_DIR / "sku_properties.joblib"
NAME_LEN            = 16
NAME_VOCAB          = 256
PRICE_BINS          = 100


def build_sku2idx(df_train_skus: pd.Series):
    """
    Map the top-(VOCAB_K-1) most frequent SKUs to indices 1..VOCAB_K-1.
    Index 0 is reserved for padding / unknown items.

    Returns (sku2idx, counts) where counts[idx] is the training-set interaction
    count of the SKU at that index (counts[0] = 0, padding).
    """
    counts_series = (
        df_train_skus
        .dropna()
        .astype(int)
        .value_counts()
        .head(VOCAB_K - 1)
    )
    skus   = counts_series.index.tolist()
    values = counts_series.values

    sku2idx = {sku: idx + 1 for idx, sku in enumerate(skus)}
    counts  = np.zeros(VOCAB_K, dtype=np.int64)
    counts[1:1 + len(values)] = values       # index 0 (padding) stays 0
    return sku2idx, counts


def get_vocab_stats(df_train_skus: pd.Series = None):
    """Load cached (sku2idx, counts) or build and cache them."""
    if VOCAB_STATS_PATH.exists():
        d = joblib.load(VOCAB_STATS_PATH)
        return d["sku2idx"], d["counts"]
    if df_train_skus is None:
        raise RuntimeError(
            f"vocab_stats not found at {VOCAB_STATS_PATH}. "
            "Pass df_train_skus to build it."
        )
    sku2idx, counts = build_sku2idx(df_train_skus)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"sku2idx": sku2idx, "counts": counts}, VOCAB_STATS_PATH)
    joblib.dump(sku2idx, SKU2IDX_PATH)   # legacy cache for backward compat
    print(
        f"  vocab_stats saved -> {VOCAB_STATS_PATH}  "
        f"({len(sku2idx):,} SKUs, counts sum={int(counts.sum()):,})"
    )
    return sku2idx, counts


def get_sku2idx(df_train_skus: pd.Series = None) -> dict:
    """Load cached sku2idx or build and cache it."""
    return get_vocab_stats(df_train_skus)[0]


def ensure_vocab_stats() -> np.ndarray:
    """
    Return the vocab counts array, rebuilding from parquet if the cache is missing.
    Used by train.py when running with sampled softmax on a stats-less checkpoint.
    """
    if VOCAB_STATS_PATH.exists():
        return joblib.load(VOCAB_STATS_PATH)["counts"]

    print(f"  vocab_stats not found at {VOCAB_STATS_PATH}. "
          f"Rebuilding from {CLEAN_PARQUET} (one-time, ~1-2 min)...")
    df = pl.read_parquet(str(CLEAN_PARQUET), columns=["timestamp", "session_id", "sku"])
    session_starts = df.group_by("session_id").agg(
        pl.col("timestamp").min().alias("session_start")
    )
    df = df.join(session_starts, on="session_id").filter(
        pl.col("session_start") < pd.Timestamp(TRAIN_CUTOFF)
    )
    _, counts = get_vocab_stats(df.select("sku").to_series().to_pandas())
    return counts


def get_sku_properties(sku2idx: dict = None) -> dict:
    """
    Build / load per-SKU auxiliary feature tables indexed by embedding index:
        price : LongArray [VOCAB_K]       0 = PAD/unknown; real = bucket+1 (range 1..100)
        name  : LongArray [VOCAB_K, 16]   0 = PAD/unknown; real = quantized token 0..255

    Both are used as input-side features in the SessionTransformer
    (looked up via the `items` tensor at train/inference time). Cached to disk.
    """
    if SKU_PROPERTIES_PATH.exists():
        return joblib.load(SKU_PROPERTIES_PATH)
    if sku2idx is None:
        raise RuntimeError(
            f"sku_properties not found at {SKU_PROPERTIES_PATH} and sku2idx not provided."
        )

    from config import DATA_DIR
    props = pl.read_parquet(
        DATA_DIR / "product_properties.parquet",
        columns=["sku", "price", "name"],
    )

    V = VOCAB_K
    price_tbl = np.zeros(V, dtype=np.int64)            # 0 = PAD
    name_tbl  = np.zeros((V, NAME_LEN), dtype=np.int64)  # 0 = PAD

    matched = 0
    for row in props.iter_rows(named=True):
        idx = sku2idx.get(int(row["sku"]))
        if idx is None:
            continue
        # Price bucket 0..99 -> shift to 1..100 so 0 can mean PAD
        p = int(row["price"])
        if 0 <= p < PRICE_BINS:
            price_tbl[idx] = p + 1
        # Name: stringified 16-element array like "[193 102 ... ]"
        name_str = row["name"]
        if name_str:
            try:
                toks = np.fromstring(name_str.strip("[]"), sep=" ", dtype=np.int64)
                if toks.size == NAME_LEN:
                    # clip into [0, NAME_VOCAB-1] just in case
                    name_tbl[idx] = np.clip(toks, 0, NAME_VOCAB - 1)
            except Exception:
                pass
        matched += 1

    out = {"price": price_tbl, "name": name_tbl}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, SKU_PROPERTIES_PATH)
    print(
        f"  sku_properties saved -> {SKU_PROPERTIES_PATH}  "
        f"({matched:,}/{V:,} SKUs matched)"
    )
    return out


def get_sku_tokens() -> np.ndarray:
    """
    Load SVD-PQ tokens [VOCAB_K, SVDPQ_T] int16 (values in [0, SVDPQ_V-1]).
    Index 0 is PAD (all zeros). Built by ingestion/svdpq.py.
    """
    if not SVDPQ_PATH.exists():
        raise FileNotFoundError(
            f"sku_tokens not found at {SVDPQ_PATH}. "
            "Run: PYTHONPATH=. uv run python ingestion/svdpq.py"
        )
    return joblib.load(SVDPQ_PATH)


def get_cat_vocab(sku2idx: dict = None):
    """
    Load (cat2idx, cat_sku_pools) from cache or build them.

    cat2idx       : raw_category_int -> dense_index (0=PAD, 1=RARE, 2..N)
    cat_sku_pools : list[np.ndarray] indexed by dense_cat_idx;
                    each entry is a sorted array of sku embedding indices
                    (matches sku2idx values) for that category.

    Requires cat2idx.joblib built by preprocess.py and sku2idx to be available.
    """
    if CAT_SKU_POOLS_PATH.exists() and CAT2IDX_PATH.exists():
        cat2idx    = joblib.load(CAT2IDX_PATH)
        cat_pools  = joblib.load(CAT_SKU_POOLS_PATH)
        return cat2idx, cat_pools

    if not CAT2IDX_PATH.exists():
        raise RuntimeError(
            f"cat2idx not found at {CAT2IDX_PATH}. Run preprocess.py first."
        )
    if sku2idx is None:
        raise RuntimeError(
            "sku2idx required to build cat_sku_pools but was not provided."
        )

    cat2idx = joblib.load(CAT2IDX_PATH)
    n_categories = max(cat2idx.values()) + 1  # 0=PAD, 1=RARE, 2..N

    from config import DATA_DIR
    props = pl.read_parquet(
        DATA_DIR / "product_properties.parquet",
        columns=["sku", "category"],
    )

    # Build pools: dense_cat_idx -> sorted array of sku_idxs
    cat_pools: list = [np.array([], dtype=np.int64) for _ in range(n_categories)]
    rows = props.iter_rows(named=True)
    pool_builder: defaultdict = defaultdict(list)
    for row in rows:
        raw_cat = row["category"]
        raw_sku = row["sku"]
        sku_idx = sku2idx.get(int(raw_sku))
        if sku_idx is None:
            continue
        dense_cat = cat2idx.get(raw_cat, 1)   # unseen → RARE
        pool_builder[dense_cat].append(sku_idx)

    for dense_cat, idxs in pool_builder.items():
        cat_pools[dense_cat] = np.array(sorted(set(idxs)), dtype=np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cat_pools, CAT_SKU_POOLS_PATH)
    print(
        f"  cat_sku_pools saved -> {CAT_SKU_POOLS_PATH}  "
        f"({n_categories} categories, "
        f"{sum(len(p) for p in cat_pools):,} sku-category pairs)"
    )
    return cat2idx, cat_pools


class SessionDataset(Dataset):
    """
    One item = one session.

    __getitem__ returns:
        events  : LongTensor [T]   action indices (ACTION2IDX)
        items   : LongTensor [T]   sku+1 for item-bearing events; 0 otherwise
        deltas  : LongTensor [T]   temporal bin indices (first event = bin 0)
        length  : int              actual session length T (before padding)
        history : dict | None      cross-session context from prior sessions of
                                   the same user (keys: events, items, deltas,
                                   length); None for first session of a user

    Sessions shorter than min_length are excluded.
    Sessions longer than max_length are truncated (most-recent events kept).
    """

    def __init__(
        self,
        parquet_path=None,
        split: str = "train",
        max_length: int = 50,
        min_length: int = 2,
        max_sessions: int = None,
        history_window: int = HISTORY_WINDOW,
    ):
        self.max_length      = max_length
        self.min_length      = min_length
        self._history_window = history_window

        path = str(parquet_path or CLEAN_PARQUET)

        # Fast path: load from disk cache if parquet hasn't changed
        cache_file = _cache_path(split, max_length, min_length, max_sessions)
        if _cache_valid(cache_file, path):
            print(f"  Loading dataset from cache: {cache_file}")
            cached = joblib.load(cache_file)
            self._sessions               = cached["sessions"]
            self._user_session_indices   = cached["user_session_indices"]
            self._session_pos_in_user    = cached["session_pos_in_user"]
            print(f"  {len(self._sessions):,} sessions loaded from cache")
            # Load cat2idx for __getitem__ (may not exist for pre-hierarchical caches)
            self._cat2idx: dict = joblib.load(CAT2IDX_PATH) if CAT2IDX_PATH.exists() else {}
            return

        # Detect which columns are present (category/price added by preprocess.py)
        _schema_cols = pl.read_parquet(path, n_rows=1).columns
        _load_cols   = ["client_id", "timestamp", "event_type", "session_id", "sku"]
        _has_category = "category" in _schema_cols
        if _has_category:
            _load_cols += ["category"]

        df = pl.read_parquet(path, columns=_load_cols).sort(["session_id", "timestamp"])

        # Session-start times for split filtering
        train_cut = pd.Timestamp(TRAIN_CUTOFF)
        val_cut   = pd.Timestamp(VAL_CUTOFF)
        session_starts = df.group_by("session_id").agg(
            pl.col("timestamp").min().alias("session_start")
        )
        df = df.join(session_starts, on="session_id")

        train_filter = pl.col("session_start") < train_cut
        if split == "train":
            df = df.filter(train_filter)
        elif split == "val":
            df = df.filter((pl.col("session_start") >= train_cut) & (pl.col("session_start") < val_cut))
        elif split == "test":
            df = df.filter(pl.col("session_start") >= val_cut)
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        # Build / load sku->embedding-index mapping (train split only, cached to disk).
        # When split != 'train', df contains only val/test rows so train_filter yields
        # nothing -rely on the cache built during the first train run instead.
        if SKU2IDX_PATH.exists():
            sku2idx = get_sku2idx()
        elif split == "train":
            sku2idx = get_sku2idx(df.select("sku").to_series().to_pandas())
        else:
            raise RuntimeError(
                f"sku2idx cache not found at {SKU2IDX_PATH}. "
                "Run SessionDataset(split='train') first to build the vocabulary."
            )

        # Load (or build) category vocabulary if available
        if CAT2IDX_PATH.exists():
            self._cat2idx, _ = get_cat_vocab(sku2idx)
        else:
            self._cat2idx = {}

        #  Vectorised feature computation (stays in Rust/numpy), replaced with Polars because Pandas enjoys
        #  jumping back to Python code instead of staying in C

        # Action indices via polars replace
        df = df.with_columns(
            pl.col("event_type")
            .replace(ACTION2IDX, default=0)
            .cast(pl.Int32)
            .alias("action_idx")
        )

        # Item indices via join against sku2idx mapping
        sku2idx_df = pl.DataFrame(
            {"sku": list(sku2idx.keys()), "item_idx": list(sku2idx.values())},
            schema={"sku": pl.Int64, "item_idx": pl.Int32},
        )
        df = (
            df.with_columns(pl.col("sku").cast(pl.Int64, strict=False))
            .join(sku2idx_df, on="sku", how="left")
            .with_columns(pl.col("item_idx").fill_null(0))
        )

        # Temporal delta (seconds) within session -vectorised diff
        df = df.with_columns(
            pl.col("timestamp").diff().over("session_id")
            .dt.total_microseconds()
            .truediv(1_000_000)   # -> seconds as float
            .fill_null(0.0)
            .alias("delta_s")
        )

        # Convert delta_s -> bin index via numpy searchsorted (one array op)
        delta_s_np = df["delta_s"].to_numpy()
        delta_s_clipped = np.clip(delta_s_np, TEMPORAL_MIN_S, TEMPORAL_MAX_S)
        bin_indices = np.minimum(
            np.searchsorted(BIN_EDGES[1:], delta_s_clipped),
            N_TEMPORAL_BINS - 1,
        ).astype(np.int32)
        # First event in each session: delta=0 -> bin 0
        first_event_mask = df["delta_s"].to_numpy() == 0.0
        bin_indices[first_event_mask & (delta_s_np == 0.0)] = 0
        df = df.with_columns(pl.Series("delta_bin", bin_indices))

        # Map raw category ints -> dense indices via cat2idx (1=RARE for unknown/rare)
        cat2idx = self._cat2idx
        if _has_category and cat2idx:
            raw_cats = df["category"].to_list()   # null for non-item rows
            dense_cats = [
                0 if c is None else cat2idx.get(c, 1)   # 0=PAD, missing→RARE(1)
                for c in raw_cats
            ]
            df = df.with_columns(pl.Series("cat_idx", dense_cats, dtype=pl.Int32))
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("cat_idx"))

        # Group into sessions and build tensor records
        agg_exprs = [
            pl.col("client_id").first().alias("client_id"),
            pl.col("action_idx").alias("actions"),
            pl.col("item_idx").alias("items"),
            pl.col("delta_bin").alias("deltas"),
            pl.col("cat_idx").alias("categories"),
            pl.len().alias("n"),
        ]
        sessions_df = (
            df.group_by("session_id", maintain_order=True)
            .agg(agg_exprs)
            .filter(pl.col("n") >= min_length)
        )
        if max_sessions is not None:
            sessions_df = sessions_df.head(max_sessions)

        # Single Python pass: build numpy records from polars lists.
        # EOS is appended as the final token so the model learns session termination
        # from real boundaries (PvA §5.3). One slot is reserved from max_length.
        # Stored as numpy arrays (not torch tensors) so joblib cache is fast to save/load;
        # conversion to tensors happens lazily in __getitem__.
        self._sessions: List[Dict] = []
        for row in sessions_df.iter_rows(named=True):
            n_raw      = min(int(row["n"]), max_length - 1)   # reserve one slot for EOS
            actions    = row["actions"][-n_raw:]    + [EOS_IDX]
            items      = row["items"][-n_raw:]      + [0]
            deltas     = row["deltas"][-n_raw:]     + [0]
            categories = row["categories"][-n_raw:] + [0]
            n          = n_raw + 1

            self._sessions.append({
                "client_id":  int(row["client_id"]),
                "events":     np.array(actions,    dtype=np.int64),
                "items":      np.array(items,      dtype=np.int64),
                "deltas":     np.array(deltas,     dtype=np.int64),
                "categories": np.array(categories, dtype=np.int64),
                "length":     n,
            })

        # Build per-user ordered index (session_id is time-based so order is preserved)
        user_idx: dict = defaultdict(list)
        for i, s in enumerate(self._sessions):
            user_idx[s["client_id"]].append(i)
        self._user_session_indices: Dict[int, List[int]] = dict(user_idx)

        # Precompute each session's position within its user's session list
        self._session_pos_in_user: Dict[int, int] = {}
        for user_sessions in self._user_session_indices.values():
            for pos, sess_idx in enumerate(user_sessions):
                self._session_pos_in_user[sess_idx] = pos

        # Save to cache for fast reloads
        print(f"  Saving dataset cache -> {cache_file}")
        joblib.dump({
            "sessions":             self._sessions,
            "user_session_indices": self._user_session_indices,
            "session_pos_in_user":  self._session_pos_in_user,
        }, cache_file)

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, idx: int) -> Dict:
        sess       = self._sessions[idx]
        cid        = sess["client_id"]
        pos        = self._session_pos_in_user[idx]
        prior_idxs = self._user_session_indices[cid][:pos]

        if prior_idxs:
            prior_e = np.concatenate([self._sessions[i]["events"] for i in prior_idxs])
            prior_i = np.concatenate([self._sessions[i]["items"]  for i in prior_idxs])
            prior_d = np.concatenate([self._sessions[i]["deltas"] for i in prior_idxs])
            H = min(len(prior_e), self._history_window)
            history = {
                "events": torch.from_numpy(prior_e[-H:]),
                "items":  torch.from_numpy(prior_i[-H:]),
                "deltas": torch.from_numpy(prior_d[-H:]),
                "length": H,
            }
        else:
            history = None

        return {
            "client_id":  sess["client_id"],
            "events":     torch.from_numpy(sess["events"]),
            "items":      torch.from_numpy(sess["items"]),
            "deltas":     torch.from_numpy(sess["deltas"]),
            "categories": torch.from_numpy(
                sess.get("categories", np.zeros(sess["length"], dtype=np.int64))
            ),
            "length":     sess["length"],
            "history":    history,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pad a list of variable-length sessions to the longest in the batch.

    Returns dict:
        events              : LongTensor [B, T_max]
        items               : LongTensor [B, T_max]
        deltas              : LongTensor [B, T_max]
        lengths             : LongTensor [B]
        tgt_key_padding_mask: BoolTensor [B, T_max-1]  True = padded position
        history             : dict | None
            events          : LongTensor [B, H_max]
            items           : LongTensor [B, H_max]
            deltas          : LongTensor [B, H_max]
            padding_mask    : BoolTensor [B, H_max]   True = padded position
    """
    max_len = max(s["length"] for s in batch)
    B = len(batch)

    ev = torch.zeros(B, max_len, dtype=torch.long)
    it = torch.zeros(B, max_len, dtype=torch.long)
    dl = torch.zeros(B, max_len, dtype=torch.long)
    ct = torch.zeros(B, max_len, dtype=torch.long)   # categories
    ln = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        L = s["length"]
        ev[i, :L] = s["events"]
        it[i, :L] = s["items"]
        dl[i, :L] = s["deltas"]
        ct[i, :L] = s["categories"]
        ln[i]     = L

    # Padding mask for transformer input (positions T-1 since forward shifts by 1).
    # True at positions >= L for each row. Vectorized over the batch.
    mask_len = max_len - 1
    pad_mask = torch.arange(mask_len).unsqueeze(0) >= ln.unsqueeze(1)

    # History: pad to max history length in batch; None if no session has history
    histories = [s.get("history") for s in batch]
    if any(h is not None for h in histories):
        max_H = max((h["length"] for h in histories if h is not None), default=0)
        h_e    = torch.zeros(B, max_H, dtype=torch.long)
        h_i    = torch.zeros(B, max_H, dtype=torch.long)
        h_d    = torch.zeros(B, max_H, dtype=torch.long)
        h_mask = torch.ones(B, max_H, dtype=torch.bool)   # True = padding
        for b, h in enumerate(histories):
            if h is not None:
                H = h["length"]
                h_e[b, :H]    = h["events"]
                h_i[b, :H]    = h["items"]
                h_d[b, :H]    = h["deltas"]
                h_mask[b, :H] = False   # real positions are not masked
        history_out: Optional[Dict] = {
            "events": h_e, "items": h_i, "deltas": h_d, "padding_mask": h_mask,
        }
    else:
        history_out = None

    return {
        "events":               ev,
        "items":                it,
        "deltas":               dl,
        "categories":           ct,
        "lengths":              ln,
        "tgt_key_padding_mask": pad_mask,
        "history":              history_out,
    }


class InteractionGenerator:
    """
    Convenience wrapper: constructs SessionDataset + torch DataLoader.
    Used by the training script.

    Attributes:
    dataset : SessionDataset
    loader  : torch.utils.data.DataLoader
    """

    def __init__(
        self,
        split: str = "train",
        batch_size: int = 64,
        max_length: int = 50,
        min_length: int = 2,
        num_workers: int = 4,
        parquet_path=None,
        max_sessions: int = None,
        history_window: int = HISTORY_WINDOW,
    ):
        self.dataset = SessionDataset(
            parquet_path=parquet_path,
            split=split,
            max_length=max_length,
            min_length=min_length,
            max_sessions=max_sessions,
            history_window=history_window,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    def __len__(self) -> int:
        return len(self.loader)
