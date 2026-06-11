"""
generate_reference_baskets.py - Real-continuation reference baskets

Builds the real-data counterpart of generate_baskets.py for the popularity-
bias evaluation: for the same anchor users a model run selects (same anchor
table, same anchors.sample(n, random_state=seed)), the "basket" is what the
user actually did next - the positive in-vocab items of the anchor session
*after* the seed event. This mirrors the generator's task exactly: the model
is conditioned on history + seed event and emits a continuation, so the
reference is the real continuation from the same point.

Conventions matched to evaluation.baskets:
  - continuation starts strictly after the first in-vocab item-bearing event
    of the anchor session (the seed; generated sessions do not contain it)
  - items filtered to positive actions (product_buy, add_to_cart), restricted
    to the model vocabulary (the generators cannot emit non-vocab SKUs, so the
    reference plays on the same item space), deduplicated by SKU in event
    order, capped at k
  - users with an empty continuation get the rank=-1 sentinel row

Output parquet uses the basket SCHEMA with algorithm="real",
mode="reference", so it flows through evaluate_popularity_bias.py unchanged:

    PYTHONPATH=. uv run python generate_reference_baskets.py --n-users 10000 --k 20 --seeds 42,43,44,45,46
    PYTHONPATH=. uv run python evaluate_popularity_bias.py \
        --baskets output/models/real_reference/baskets/*.parquet \
        --labels real-s42 real-s43 real-s44 real-s45 real-s46
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from config import MODEL_DIR
from evaluation.baskets import SCHEMA, _build_anchor_table, save
from ingestion.dataset import get_sku2idx


POSITIVE_ACTIONS = ("product_buy", "add_to_cart")
ITEM_BEARING = ("add_to_cart", "remove_from_cart", "product_buy")
DEFAULT_OUT_DIR = MODEL_DIR / "real_reference" / "baskets"


def parse_args():
    p = argparse.ArgumentParser(description="Generate real-continuation reference baskets")
    p.add_argument("--n-users", type=int, default=10000)
    p.add_argument("--k", type=int, default=20, help="Max items per reference basket")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46",
                   help="Comma-separated seeds; each yields one parquet with the "
                        "same users as the model basket run of that seed")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def _continuation_rows(anchor_events: pd.DataFrame, vocab: set, k: int) -> list[tuple]:
    """Real-continuation basket rows for one user's anchor-session events."""
    user_id = int(anchor_events["client_id"].iloc[0])
    events = anchor_events.sort_values("timestamp", kind="stable")

    is_seedable = (
        events["event_type"].isin(ITEM_BEARING)
        & events["sku_int"].notna()
        & events["sku_int"].isin(vocab)
    )
    if not is_seedable.any():
        # cannot happen for anchor sessions by construction, but stay safe
        return [(user_id, -1, None, None, None, "real", "reference")]
    seed_pos = is_seedable.to_numpy().argmax()
    cont = events.iloc[seed_pos + 1:]

    cont = cont[
        cont["event_type"].isin(POSITIVE_ACTIONS)
        & cont["sku_int"].notna()
        & cont["sku_int"].isin(vocab)
    ]

    rows: list[tuple] = []
    seen: set = set()
    for ev in cont.itertuples(index=False):
        sku = int(ev.sku_int)
        if sku in seen:
            continue
        seen.add(sku)
        rows.append((user_id, len(rows), sku, None, ev.timestamp, "real", "reference"))
        if len(rows) >= k:
            break
    if not rows:
        rows.append((user_id, -1, None, None, None, "real", "reference"))
    return rows


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    t0 = time.time()

    print("Building anchor table (shared with generate_baskets.py) ...")
    anchors, _session_meta, train_events = _build_anchor_table()
    if len(anchors) < args.n_users:
        raise ValueError(
            f"requested {args.n_users:,} users but only {len(anchors):,} eligible"
        )
    vocab = set(get_sku2idx().keys())

    anchor_keys = anchors[["client_id", "session_id"]]
    events_pd = (
        train_events
        .select(["client_id", "session_id", "timestamp", "event_type", "sku_int"])
        .to_pandas()
        .merge(anchor_keys, on=["client_id", "session_id"])
    )

    for seed in seeds:
        selected = anchors.sample(n=args.n_users, random_state=seed)
        sel_keys = selected[["client_id", "session_id"]]
        sel_events = events_pd.merge(sel_keys, on=["client_id", "session_id"])

        rows: list[tuple] = []
        for _, user_events in sel_events.groupby("client_id", sort=False):
            rows.extend(_continuation_rows(user_events, vocab, args.k))

        df = pd.DataFrame(rows, columns=SCHEMA)
        df["item_id"] = df["item_id"].astype("Int64")

        out_path = out_dir / f"baskets-real-s{seed}.parquet"
        save(df, out_path)
        out_path.with_suffix(".config.yaml").write_text(yaml.safe_dump({
            "mode": "reference",
            "algorithm": "real",
            "n_users": args.n_users,
            "k": args.k,
            "seed": seed,
            "output": str(out_path),
        }, sort_keys=False))

        with_basket = df[df["rank"] >= 0]
        sizes = with_basket.groupby("user_id")["rank"].count()
        print(f"  seed {seed}: {sizes.size:,}/{args.n_users:,} users with non-empty "
              f"continuation, mean size {sizes.mean():.2f} -> {out_path.name}")

    print(f"Done in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
