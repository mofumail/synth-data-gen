"""
Synthetic data exploration / failure diagnostics.

Ad-hoc companion to FidelityEvaluator, prints readable stats
about a synthetic sessions parquet and compares them to the real raw data and
training distribution.

Usage:
    uv run --with polars --with pyarrow python -m evaluation.synthetic_data_evaluation \\
        output/synthetic/session_transformer_d128_l4_h4_svdpq-seed42-1000000.parquet

    # Compare two synthetic runs side by side
    uv run --with polars --with pyarrow python -m evaluation.synthetic_data_evaluation \\
        output/synthetic/session_transformer_d128_l4_h4_svdpq-seed42-1000000.parquet \\
        output/synthetic/session_transformer_d128_l4_h4_hier-seed42-1000000.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from config import DATA_DIR, CLEAN_PARQUET


ITEM_EVENTS = ("add_to_cart", "remove_from_cart", "product_buy")


def _popularity_shares(df: pl.DataFrame, ks=(20, 100, 1000)) -> dict:
    vc = df.group_by("sku").len().sort("len", descending=True)
    total = df.height
    out = {"unique_skus": len(vc)}
    for k in ks:
        out[f"top{k}_share"] = vc.head(k)["len"].sum() / total if total else 0.0
    out["_top100_skus"] = set(vc.head(100)["sku"].cast(pl.Int64).to_list())
    return out


def summarise(df: pl.DataFrame, label: str) -> dict:
    print(f"\n {label} ")
    print(f"rows: {df.height:,}")

    if "session_id" in df.columns:
        sl = df.group_by("session_id").len()["len"]
        print(f"sessions: {df['session_id'].n_unique():,}  "
              f"len mean={sl.mean():.1f} median={sl.median():.0f} max={sl.max()}")

    counts = df.group_by("event_type").len().sort("len", descending=True)
    print("event counts:")
    for row in counts.iter_rows():
        print(f"  {row[0]:<18s} {row[1]:>12,}")

    def n(et: str) -> int:
        return df.filter(pl.col("event_type") == et).height

    n_buy, n_atc, n_rmc = n("product_buy"), n("add_to_cart"), n("remove_from_cart")
    if n_atc > 0:
        print(f"buy/atc: {n_buy/n_atc:.4f}    rmc/atc: {n_rmc/n_atc:.4f}")

    if "session_id" in df.columns:
        per_sess = (
            df.group_by("session_id")
              .agg((pl.col("event_type") == "product_buy").sum().alias("n_buy"))
        )
        print(f"sessions with >=1 buy: {(per_sess['n_buy'] >= 1).mean():.3%}")

    atc = df.filter(pl.col("event_type") == "add_to_cart")
    pop = _popularity_shares(atc) if atc.height else {}
    if pop:
        print(f"atc unique skus: {pop['unique_skus']:,}")
        print(f"atc top-20  share: {pop['top20_share']:.3%}")
        print(f"atc top-100 share: {pop['top100_share']:.3%}")
        print(f"atc top-1000 share: {pop['top1000_share']:.3%}")

    if "sku" in df.columns:
        item = df.filter(pl.col("sku").is_not_null())
        print(f"all item events: {item.height:,}  unique skus: {item['sku'].n_unique():,}")

    return {"label": label, "n_buy": n_buy, "n_atc": n_atc, "n_rmc": n_rmc, "atc_pop": pop}


def load_real_reference() -> dict:
    """Load raw add-to-cart for an apples-to-apples popularity baseline."""
    atc = pl.read_parquet(DATA_DIR / "add_to_cart.parquet").select(["sku"])
    pb  = pl.read_parquet(DATA_DIR / "product_buy.parquet").select(["sku"])
    rc  = pl.read_parquet(DATA_DIR / "remove_from_cart.parquet").select(["sku"])
    return {
        "atc_pop": _popularity_shares(atc),
        "buy_atc_ratio": pb.height / atc.height,
        "rmc_atc_ratio": rc.height / atc.height,
        "n_buy": pb.height, "n_atc": atc.height, "n_rmc": rc.height,
    }


def load_training_reference() -> pl.DataFrame:
    return pl.read_parquet(CLEAN_PARQUET)


def compare(synth_summaries: list[dict], real_ref: dict, train_summary: dict):
    print("\n\n=== side-by-side ===")
    print(f"{'metric':<30s}{'real':>14s}{'training':>14s}", end="")
    for s in synth_summaries:
        print(f"{s['label'][:14]:>14s}", end="")
    print()

    def row(name, real_v, train_v, synth_fn, fmt="{:.4f}"):
        print(f"{name:<30s}{fmt.format(real_v):>14s}{fmt.format(train_v):>14s}", end="")
        for s in synth_summaries:
            try:
                v = synth_fn(s)
                print(f"{fmt.format(v):>14s}", end="")
            except Exception:
                print(f"{'—':>14s}", end="")
        print()

    row("buy/atc",
        real_ref["buy_atc_ratio"],
        train_summary["n_buy"] / max(train_summary["n_atc"], 1),
        lambda s: s["n_buy"] / max(s["n_atc"], 1))
    row("rmc/atc",
        real_ref["rmc_atc_ratio"],
        train_summary["n_rmc"] / max(train_summary["n_atc"], 1),
        lambda s: s["n_rmc"] / max(s["n_atc"], 1))
    row("atc top-20 share",
        real_ref["atc_pop"]["top20_share"],
        train_summary["atc_pop"]["top20_share"],
        lambda s: s["atc_pop"]["top20_share"])
    row("atc top-100 share",
        real_ref["atc_pop"]["top100_share"],
        train_summary["atc_pop"]["top100_share"],
        lambda s: s["atc_pop"]["top100_share"])
    row("atc top-1000 share",
        real_ref["atc_pop"]["top1000_share"],
        train_summary["atc_pop"]["top1000_share"],
        lambda s: s["atc_pop"]["top1000_share"])
    row("atc unique skus",
        real_ref["atc_pop"]["unique_skus"],
        train_summary["atc_pop"]["unique_skus"],
        lambda s: s["atc_pop"]["unique_skus"],
        fmt="{:,.0f}")

    real_top100 = real_ref["atc_pop"]["_top100_skus"]
    print(f"{'top-100 overlap w/ real':<30s}{'—':>14s}"
          f"{len(real_top100 & train_summary['atc_pop']['_top100_skus']):>14d}", end="")
    for s in synth_summaries:
        print(f"{len(real_top100 & s['atc_pop']['_top100_skus']):>14d}", end="")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synthetic", nargs="+", type=Path,
                    help="One or more synthetic-sessions parquet files.")
    ap.add_argument("--skip-real", action="store_true",
                    help="Skip loading raw ../DATA/*.parquet baselines (faster).")
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip loading events_clean.parquet baseline (faster).")
    args = ap.parse_args()

    synth_summaries = []
    for path in args.synthetic:
        df = pl.read_parquet(path)
        synth_summaries.append(summarise(df, label=path.stem))

    if not args.skip_training:
        train_df = load_training_reference()
        train_summary = summarise(train_df, label="events_clean (train)")
    else:
        train_summary = None

    if not args.skip_real and not args.skip_training:
        real_ref = load_real_reference()
        compare(synth_summaries, real_ref, train_summary)


if __name__ == "__main__":
    main()
