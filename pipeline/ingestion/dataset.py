"""
SessionDataset + InteractionGenerator

PyTorch Dataset wrapping events_clean.parquet for SessionTransformer training.

Each item is one session: (events, codes, deltas) tensors used as input to
SessionTransformer.forward(). The forward() method internally shifts by 1
(input t=0..T-2, target t=1..T-1), so we store the full session here.

Items are represented as RQ-VAE codes: each event has N_CODE_LEVELS=3 code
indices from a codebook of size RQVAE_CODEBOOK_SIZE=128. Run
`uv run python ingestion/rqvae.py` first to generate sku2codes.joblib.

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
    CLEAN_PARQUET, HISTORY_WINDOW, OUTPUT_DIR, TRAIN_CUTOFF, VAL_CUTOFF,
    SKU2CODES_PATH, RQVAE_N_LEVELS,
)
from simulation.generator.session_transformer import (
    ACTION2IDX, BIN_EDGES, EOS_IDX, N_TEMPORAL_BINS, TEMPORAL_MIN_S, TEMPORAL_MAX_S,
)

N_CODE_LEVELS = RQVAE_N_LEVELS   # 3


def _cache_path(split: str, max_length: int, min_length: int, max_sessions) -> str:
    tag = f"{split}_ml{max_length}_min{min_length}_ms{max_sessions or 'all'}_rqvae"
    return str(OUTPUT_DIR / f"session_cache_{tag}.joblib")


def _cache_valid(cache_file: str, parquet_path: str) -> bool:
    """Cache is valid if it exists and is newer than the source parquet."""
    if not os.path.exists(cache_file):
        return False
    return os.path.getmtime(cache_file) >= os.path.getmtime(parquet_path)


def get_sku2codes() -> tuple:
    """
    Load (sku2codes, codes2sku) from SKU2CODES_PATH.
    Raises RuntimeError if the file is not found — run ingestion/rqvae.py first.
    """
    if not SKU2CODES_PATH.exists():
        raise RuntimeError(
            f"sku2codes not found at {SKU2CODES_PATH}. "
            "Run `uv run python ingestion/rqvae.py` first to train the RQ-VAE "
            "and build the item code mapping."
        )
    return joblib.load(SKU2CODES_PATH)


class SessionDataset(Dataset):
    """
    One item = one session.

    __getitem__ returns:
        events  : LongTensor [T]     action indices (ACTION2IDX)
        codes   : LongTensor [T, 3]  RQ-VAE item codes; (0,0,0) for non-item events
        deltas  : LongTensor [T]     temporal bin indices (first event = bin 0)
        length  : int                actual session length T (before padding)
        history : dict | None        cross-session context from prior sessions of
                                     the same user (keys: events, codes, deltas,
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
            return

        df = pl.read_parquet(
            path,
            columns=["client_id", "timestamp", "event_type", "session_id", "sku"],
        ).sort(["session_id", "timestamp"])

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

        # Load RQ-VAE item code mapping (built by ingestion/rqvae.py)
        sku2codes, _ = get_sku2codes()

        # Action indices via polars replace
        df = df.with_columns(
            pl.col("event_type")
            .replace(ACTION2IDX, default=0)
            .cast(pl.Int32)
            .alias("action_idx")
        )

        # RQ-VAE code indices via join against sku2codes mapping
        codes_df = pl.DataFrame(
            {
                "sku": list(sku2codes.keys()),
                "c0":  [int(v[0]) for v in sku2codes.values()],
                "c1":  [int(v[1]) for v in sku2codes.values()],
                "c2":  [int(v[2]) for v in sku2codes.values()],
            },
            schema={"sku": pl.Int64, "c0": pl.Int32, "c1": pl.Int32, "c2": pl.Int32},
        )
        df = (
            df.with_columns(pl.col("sku").cast(pl.Int64, strict=False))
            .join(codes_df, on="sku", how="left")
            .with_columns([
                pl.col("c0").fill_null(0),
                pl.col("c1").fill_null(0),
                pl.col("c2").fill_null(0),
            ])
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

        # Group into sessions and build tensor records
        sessions_df = (
            df.group_by("session_id", maintain_order=True)
            .agg(
                pl.col("client_id").first().alias("client_id"),
                pl.col("action_idx").alias("actions"),
                pl.col("c0").alias("c0_list"),
                pl.col("c1").alias("c1_list"),
                pl.col("c2").alias("c2_list"),
                pl.col("delta_bin").alias("deltas"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_length)
        )
        if max_sessions is not None:
            sessions_df = sessions_df.head(max_sessions)

        # Single Python pass: build numpy records from polars lists.
        # EOS is appended as the final token so the model learns session termination.
        # Codes for EOS token = (0, 0, 0) (padding triple).
        self._sessions: List[Dict] = []
        for row in sessions_df.iter_rows(named=True):
            n_raw   = min(int(row["n"]), max_length - 1)   # reserve one slot for EOS
            actions = row["actions"][-n_raw:] + [EOS_IDX]
            c0      = row["c0_list"][-n_raw:] + [0]
            c1      = row["c1_list"][-n_raw:] + [0]
            c2      = row["c2_list"][-n_raw:] + [0]
            deltas  = row["deltas"][-n_raw:]  + [0]
            n       = n_raw + 1
            # codes: [T, N_CODE_LEVELS] array
            codes   = np.stack([c0, c1, c2], axis=1).astype(np.int32)

            self._sessions.append({
                "client_id": int(row["client_id"]),
                "events":    np.array(actions, dtype=np.int64),
                "codes":     codes,                            # [T, 3]
                "deltas":    np.array(deltas,  dtype=np.int64),
                "length":    n,
            })

        # Build per-user ordered index
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
            prior_c = np.concatenate([self._sessions[i]["codes"]  for i in prior_idxs])  # [H_all, 3]
            prior_d = np.concatenate([self._sessions[i]["deltas"] for i in prior_idxs])
            H = min(len(prior_e), self._history_window)
            history = {
                "events": torch.from_numpy(prior_e[-H:]),
                "codes":  torch.from_numpy(prior_c[-H:]),   # [H, 3]
                "deltas": torch.from_numpy(prior_d[-H:]),
                "length": H,
            }
        else:
            history = None

        return {
            "client_id": sess["client_id"],
            "events":    torch.from_numpy(sess["events"]),
            "codes":     torch.from_numpy(sess["codes"]),   # [T, 3]
            "deltas":    torch.from_numpy(sess["deltas"]),
            "length":    sess["length"],
            "history":   history,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pad a list of variable-length sessions to the longest in the batch.

    Returns dict:
        events              : LongTensor [B, T_max]
        codes               : LongTensor [B, T_max, 3]
        deltas              : LongTensor [B, T_max]
        lengths             : LongTensor [B]
        tgt_key_padding_mask: BoolTensor [B, T_max-1]  True = padded position
        history             : dict | None
            events          : LongTensor [B, H_max]
            codes           : LongTensor [B, H_max, 3]
            deltas          : LongTensor [B, H_max]
            padding_mask    : BoolTensor [B, H_max]   True = padded position
    """
    max_len = max(s["length"] for s in batch)
    B = len(batch)

    ev = torch.zeros(B, max_len, dtype=torch.long)
    co = torch.zeros(B, max_len, N_CODE_LEVELS, dtype=torch.long)
    dl = torch.zeros(B, max_len, dtype=torch.long)
    ln = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        L = s["length"]
        ev[i, :L]    = s["events"]
        co[i, :L, :] = s["codes"]
        dl[i, :L]    = s["deltas"]
        ln[i]        = L

    # Padding mask for transformer input (positions T-1 since forward shifts by 1)
    mask_len = max_len - 1
    pad_mask = torch.zeros(B, mask_len, dtype=torch.bool)
    for i, s in enumerate(batch):
        L = s["length"]
        if L < mask_len:
            pad_mask[i, L:] = True

    # History: pad to max history length in batch; None if no session has history
    histories = [s.get("history") for s in batch]
    if any(h is not None for h in histories):
        max_H  = max((h["length"] for h in histories if h is not None), default=0)
        h_e    = torch.zeros(B, max_H, dtype=torch.long)
        h_c    = torch.zeros(B, max_H, N_CODE_LEVELS, dtype=torch.long)
        h_d    = torch.zeros(B, max_H, dtype=torch.long)
        h_mask = torch.ones(B, max_H, dtype=torch.bool)   # True = padding
        for b, h in enumerate(histories):
            if h is not None:
                H = h["length"]
                h_e[b, :H]       = h["events"]
                h_c[b, :H, :]    = h["codes"]
                h_d[b, :H]       = h["deltas"]
                h_mask[b, :H]    = False   # real positions are not masked
        history_out: Optional[Dict] = {
            "events": h_e, "codes": h_c, "deltas": h_d, "padding_mask": h_mask,
        }
    else:
        history_out = None

    return {
        "events":               ev,
        "codes":                co,
        "deltas":               dl,
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
