"""
Recommendation-basket generation for downstream fairness / beyond-accuracy
evaluation

The default path samples unique real training users, preloads their
prior-session history, and reads the model's top-K item distribution from that
context. Rollout mode is for sampled behavioral sessions.

Storage: long-form parquet on disk (user_id - rank - item_id - score -
timestamp - algorithm - mode) , joinable with item-side tables for the fairness
metrics. Use load_as_dict() to round-trip into {user_id: [item_id, ...]}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import polars as pl
import torch

from config import (
    CLEAN_PARQUET, EVAL_MODEL_NAME, EVAL_MODEL_SUBDIR, HISTORY_WINDOW,
    ITEM_BEARING_EVENTS, MODEL_DIR, OUTPUT_DIR, TRAIN_CUTOFF, VOCAB_K,
)
from ingestion.dataset import get_sku2idx
from evaluation.reference import ReferenceStore
from simulation.generator.session_generator import SessionGenerator
from simulation.validity import ValidityLayer


DEFAULT_BASKET_ACTIONS: tuple[str, ...] = ("product_buy", "add_to_cart")
DEFAULT_INPUT_SOURCE = "train"
REF_STORE_PATH = OUTPUT_DIR / f"reference_store_v{VOCAB_K}.joblib"


def _build_validity_layer() -> tuple[ValidityLayer, dict]:
    """
    Build a ValidityLayer from the cached training ReferenceStore so basket
    rollouts are filtered to the same legal action bigrams as evaluate.py.

    Falls back to an empty bigram set if the cache is missing, monotonicity
    and purchase-exposure checks still apply, but illegal-transition filtering
    is disabled.
    """
    from collections import defaultdict
    if REF_STORE_PATH.exists():
        rs = ReferenceStore.load(REF_STORE_PATH)
        valid_transitions: dict = defaultdict(set)
        for a, b in rs.legal_bigrams:
            valid_transitions[a].add(b)
        return ValidityLayer(legal_bigrams=rs.legal_bigrams_set()), valid_transitions
    print(
        f"  [warn] ReferenceStore cache not found at {REF_STORE_PATH}; "
        f"illegal-bigram check disabled. Run evaluate.py once to populate it."
    )
    return ValidityLayer(legal_bigrams=set()), {}


def _build_generator(
    model_path: Path,
    batch_size: int,
    identity_sampler_path: Optional[Path],
) -> SessionGenerator:
    validity_layer, valid_transitions = _build_validity_layer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SessionGenerator(
        model_path            = str(model_path),
        validity_layer        = validity_layer,
        valid_transitions     = valid_transitions,
        identity_sampler_path = str(identity_sampler_path) if identity_sampler_path else None,
        device                = device,
        batch_size            = batch_size,
    )


def _events_to_basket(
    events: Sequence[dict],
    basket_actions: Iterable[str],
    k: int,
) -> list[dict]:
    """Filter, dedup-by-sku, truncate to k. Preserves emission order."""
    keep_actions = set(basket_actions)
    seen: set = set()
    out: list[dict] = []
    for ev in events:
        if ev.get("event_type") not in keep_actions:
            continue
        sku = ev.get("sku")
        if sku is None or sku in seen:
            continue
        seen.add(sku)
        out.append(ev)
        if len(out) >= k:
            break
    return out


SCHEMA = ["user_id", "rank", "item_id", "score", "timestamp", "algorithm", "mode"]


def _read_train_events() -> pl.DataFrame:
    """Load train-split events with session starts, matching SessionDataset."""
    columns = ["client_id", "timestamp", "event_type", "session_id", "sku"]
    df = pl.read_parquet(str(CLEAN_PARQUET), columns=columns)
    session_starts = df.group_by(["client_id", "session_id"]).agg(
        pl.col("timestamp").min().alias("session_start")
    )
    return (
        df.join(session_starts, on=["client_id", "session_id"])
        .filter(pl.col("session_start") < pd.Timestamp(TRAIN_CUTOFF))
        .with_columns(pl.col("sku").cast(pl.Int64, strict=False).alias("sku_int"))
        .sort(["client_id", "session_start", "timestamp", "session_id"])
    )


def _build_train_user_inputs(n_users: int, seed: int) -> list[tuple]:
    """
    Build one basket input per unique real training user.

    For each selected user, anchor on their latest train session that contains an
    in-vocab item-bearing event. The seed SKU is the first such item in that
    anchor session, cross-session history is all earlier sessions, truncated to
    HISTORY_WINDOW after adding EOS separators.
    """
    print("  Building unique real-user basket inputs from train history ...")
    sku2idx = get_sku2idx()
    vocab_skus = list(sku2idx.keys())
    df = _read_train_events()

    session_meta = (
        df.select(["client_id", "session_id", "session_start"])
        .unique()
        .sort(["client_id", "session_start", "session_id"])
        .to_pandas()
    )
    session_meta["session_pos"] = session_meta.groupby("client_id", sort=False).cumcount()

    item_sessions = (
        df.filter(
            pl.col("event_type").is_in(list(ITEM_BEARING_EVENTS))
            & pl.col("sku_int").is_not_null()
            & pl.col("sku_int").is_in(vocab_skus)
        )
        .group_by(["client_id", "session_id"], maintain_order=True)
        .agg(
            pl.col("sku_int").first().alias("seed_sku"),
            pl.col("session_start").first().alias("session_start"),
        )
        .to_pandas()
    )

    anchors = item_sessions.merge(
        session_meta[["client_id", "session_id", "session_start", "session_pos"]],
        on=["client_id", "session_id", "session_start"],
        how="left",
    )
    anchors = anchors[anchors["session_pos"] > 0]
    anchors = (
        anchors.sort_values(["client_id", "session_start", "session_id"])
        .groupby("client_id", sort=False)
        .tail(1)
    )
    if len(anchors) < n_users:
        raise ValueError(
            f"requested {n_users:,} unique users with train history, but only "
            f"{len(anchors):,} eligible users are available"
        )

    selected = anchors.sample(n=n_users, random_state=seed).reset_index(drop=True)
    selected_ids = selected["client_id"].astype(int).tolist()
    selected_by_cid = {
        int(row.client_id): row
        for row in selected.itertuples(index=False)
    }

    events_pd = (
        df.filter(pl.col("client_id").is_in(selected_ids))
        .select(["client_id", "session_id", "session_start", "timestamp", "event_type", "sku_int"])
        .to_pandas()
        .merge(
            session_meta[["client_id", "session_id", "session_pos"]],
            on=["client_id", "session_id"],
            how="left",
        )
        .sort_values(["client_id", "session_pos", "timestamp", "session_id"])
    )

    inputs: list[tuple] = []
    for cid, user_events in events_pd.groupby("client_id", sort=False):
        cid = int(cid)
        anchor = selected_by_cid[cid]
        prior = user_events[user_events["session_pos"] < int(anchor.session_pos)]
        history: list[dict] = []
        for _, session in prior.groupby("session_id", sort=False):
            session = session.sort_values("timestamp")
            for ev in session.itertuples(index=False):
                sku = None if pd.isna(ev.sku_int) else int(ev.sku_int)
                history.append({
                    "event_type": str(ev.event_type),
                    "sku": sku,
                    "timestamp": ev.timestamp,
                })
            history.append({"event_type": "EOS", "sku": None, "timestamp": None})

        hist = history[-HISTORY_WINDOW:] if history else None
        inputs.append((cid, int(anchor.seed_sku), pd.Timestamp(anchor.session_start), hist))

    if len(inputs) != n_users:
        raise RuntimeError(
            f"built {len(inputs):,} basket inputs for {n_users:,} selected users"
        )
    return inputs


def generate_baskets(
    model_path: Path,
    n_users: int,
    k: int = 20,
    seed: int = 42,
    basket_actions: Iterable[str] = DEFAULT_BASKET_ACTIONS,
    *,
    mode: str = "topk",
    input_source: str = DEFAULT_INPUT_SOURCE,
    rec_action: Optional[str] = None,
    top_c: int = 100,
    batch_size: int = 512,
    identity_sampler_path: Optional[Path] = None,
    algorithm: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate per-user recommendation baskets from the trained SessionTransformer.

    input_source="train" (default): sample unique real train users and preload
        their prior-session history.
    input_source="sampler": use the legacy identity sampler/random fallback;
        score_baskets still enforces unique user IDs.
    mode="topk" (default): read the item-head distribution at step 0 and take
        the k most probable SKUs per user, conditioned on the required
        rec_action. Exactly k items, ranked by score.
    mode="rollout": sample a behavioral session per user and filter events to
        basket_actions..

    Returns a long-form DataFrame with columns:
        user_id - rank - item_id - score -timestamp - algorithm - mode
    One row per (user, basket position). In rollout mode, users with zero
    basket items get a sentinel rank=-1 / item_id=NA row so they survive the
    dict round-trip.
    """
    if mode not in {"topk", "rollout"}:
        raise ValueError(f"mode must be 'topk' or 'rollout', got {mode!r}")
    if input_source not in {"train", "sampler"}:
        raise ValueError(f"input_source must be 'train' or 'sampler', got {input_source!r}")
    if mode == "topk" and not rec_action:
        raise ValueError("rec_action is required when mode='topk'")
    if algorithm is None:
        algorithm = EVAL_MODEL_NAME or "unknown"

    gen = _build_generator(model_path, batch_size, identity_sampler_path)
    batch_inputs = _build_train_user_inputs(n_users, seed) if input_source == "train" else None

    if mode == "topk":
        rows, short_count, empty_count = _topk_rows(
            gen, n_users, seed, k, rec_action, top_c, algorithm, batch_inputs
        )
    else:
        rows, short_count, empty_count = _rollout_rows(
            gen, n_users, seed, k, basket_actions, algorithm, batch_inputs
        )

    df = pd.DataFrame(rows, columns=SCHEMA)
    # Nullable Int64 so the sentinel (empty-basket) rows don't coerce SKUs to float.
    df["item_id"] = df["item_id"].astype("Int64")
    df.attrs["n_users_requested"]   = n_users
    df.attrs["n_users_with_basket"] = int((df["rank"] >= 0).groupby(df["user_id"]).any().sum())
    df.attrs["short_baskets"]       = short_count
    df.attrs["empty_baskets"]       = empty_count
    df.attrs["k"]                   = k
    df.attrs["mode"]                = mode
    df.attrs["input_source"]        = input_source
    return df


def _topk_rows(gen, n_users, seed, k, rec_action, top_c, algorithm, batch_inputs=None):
    """Top-K scored baskets -> rows. timestamp is NA (no event time)."""
    if batch_inputs is None:
        baskets = gen.score_baskets(
            n_users=n_users, seed=seed, k=k, rec_action=rec_action, top_c=top_c
        )
        user_ids = [basket[0]["client_id"] if basket else None for basket in baskets]
    else:
        baskets = gen.score_baskets_for_inputs(
            batch_inputs, k=k, rec_action=rec_action, top_c=top_c
        )
        user_ids = [bi[0] for bi in batch_inputs]

    rows: list[tuple] = []
    short_count = empty_count = 0
    for idx, basket in enumerate(baskets):
        if not basket:
            user_id = user_ids[idx] if idx < len(user_ids) else None
            if user_id is not None:
                rows.append((user_id, -1, None, None, None, algorithm, "topk"))
            empty_count += 1
            continue
        user_id = basket[0]["client_id"]
        if len(basket) < k:
            short_count += 1
        for item in basket:
            rows.append(
                (user_id, item["rank"], item["sku"], item["score"], None, algorithm, "topk")
            )
    return rows, short_count, empty_count


def _rollout_rows(gen, n_users, seed, k, basket_actions, algorithm, batch_inputs=None):
    """Sampled-session baskets -> rows. score is NA (sampled, not scored)."""
    if batch_inputs is None:
        sessions = gen.generate(n_sessions=n_users, seed=seed, apply_constraints=True)
        user_ids = [session[0].get("client_id") if session else None for session in sessions]
    else:
        sessions = gen.generate_from_inputs(batch_inputs, seed=seed, apply_constraints=True)
        user_ids = [bi[0] for bi in batch_inputs]

    rows: list[tuple] = []
    short_count = empty_count = 0
    for idx, session in enumerate(sessions):
        user_id = user_ids[idx] if idx < len(user_ids) else None
        if not session:
            if user_id is not None:
                rows.append((user_id, -1, None, None, None, algorithm, "rollout"))
            empty_count += 1
            continue
        user_id = session[0].get("client_id")
        basket = _events_to_basket(session, basket_actions, k)
        if not basket:
            rows.append((user_id, -1, None, None, None, algorithm, "rollout"))
            empty_count += 1
            continue
        if len(basket) < k:
            short_count += 1
        for rank, ev in enumerate(basket):
            rows.append(
                (user_id, rank, ev["sku"], None, ev["timestamp"], algorithm, "rollout")
            )
    return rows, short_count, empty_count


def save(df: pd.DataFrame, path: Path) -> None:
    """Write the basket DataFrame to parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_as_dict(path: Path) -> dict[Any, list]:
    """
    Read a basket parquet and return {user_id: [item_id, …]} ordered by
    rank.

    Users whose row has rank=-1 (empty basket) appear with an empty list.
    """
    df = pd.read_parquet(path)
    out: dict[Any, list] = {}
    for user_id, grp in df.groupby("user_id", sort=False):
        valid = grp[grp["rank"] >= 0].sort_values("rank")
        out[user_id] = valid["item_id"].tolist()
    return out
