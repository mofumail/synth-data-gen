"""
Merges the 6 raw Parquet files, deduplicates, sessionizes, and saves two Parquets:
  - events_clean.parquet : train + val sessions (timestamp < VAL_CUTOFF)
  - events_test.parquet  : test sessions (timestamp >= VAL_CUTOFF), locked

Output columns:
    client_id   : int64
    timestamp   : datetime[us] (tz-naive UTC)
    event_type  : str
    session_id  : int32
    sku         : float64 (null for non-item-bearing events)
"""

import polars as pl
import pyarrow.parquet as pq
import pandas as pd

from config import (
    DATA_DIR, OUTPUT_DIR, CLEAN_PARQUET, TEST_PARQUET,
    DS_START, DS_END, VAL_CUTOFF, SESSION_TIMEOUT_MIN, PAGE_VISIT_SAMPLE,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert config date strings/objects to Python datetime for polars comparisons
_DS_START = pd.Timestamp(DS_START).to_pydatetime()
_DS_END   = pd.Timestamp(DS_END).to_pydatetime()


def _normalize_ts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Parse/strip timezone, yield tz-naive Datetime[us]."""
    ts_dtype = lf.collect_schema()["timestamp"]
    if ts_dtype == pl.Utf8 or ts_dtype == pl.String:
        # String timestamps, parse directly (Synerise format: "YYYY-MM-DD HH:MM:SS")
        lf = lf.with_columns(
            pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
        )
    elif getattr(ts_dtype, "time_zone", None) is not None:
        # Timezone-aware datetime, convert to UTC then strip tz
        lf = lf.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    # Ensure us precision (may already be correct after parsing)
    return lf.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def _filter_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        (pl.col("timestamp") >= _DS_START) & (pl.col("timestamp") <= _DS_END)
    )


# Synerise has logging retry errors, same (client_id, timestamp, sku) can appear multiple times
def load_item_bearing(name: str) -> pl.DataFrame:
    lf = pl.scan_parquet(DATA_DIR / f"{name}.parquet").select(["client_id", "timestamp", "sku"])
    lf = _normalize_ts(lf)
    lf = _filter_dates(lf)
    df = lf.collect()
    before = df.height
    df = df.unique(subset=["client_id", "timestamp", "sku"])
    dupes = before - df.height
    df = df.with_columns([
        pl.lit(name).alias("event_type"),
        pl.col("sku").cast(pl.Float64),
    ]).select(["client_id", "timestamp", "event_type", "sku"])
    print(f"  {name:<25} {df.height:>10,} rows  ({dupes:,} dupes removed)")
    return df


def load_search_query() -> pl.DataFrame:
    lf = pl.scan_parquet(DATA_DIR / "search_query.parquet").select(["client_id", "timestamp"])
    lf = _normalize_ts(lf)
    lf = _filter_dates(lf)
    df = lf.collect()
    before = df.height
    df = df.unique(subset=["client_id", "timestamp"])
    dupes = before - df.height
    df = df.with_columns([
        pl.lit("search_query").alias("event_type"),
        pl.lit(None).cast(pl.Float64).alias("sku"),
    ]).select(["client_id", "timestamp", "event_type", "sku"])
    print(f"  {'search_query':<25} {df.height:>10,} rows  ({dupes:,} dupes removed)")
    return df


def load_page_visit() -> pl.DataFrame:
    path = DATA_DIR / "page_visit.parquet"
    total_rows = pq.ParquetFile(path).metadata.num_rows

    lf = pl.scan_parquet(path).select(["client_id", "timestamp"])
    lf = _normalize_ts(lf)
    lf = _filter_dates(lf)

    if total_rows > PAGE_VISIT_SAMPLE:
        df = lf.collect()
        df = df.sample(n=PAGE_VISIT_SAMPLE, seed=42)
        print(f"  {'page_visit':<25} {df.height:>10,} rows  (sampled {PAGE_VISIT_SAMPLE:,} from {total_rows:,})")
    else:
        df = lf.collect()
        print(f"  {'page_visit':<25} {df.height:>10,} rows  (all {total_rows:,} rows)")

    df = df.with_columns([
        pl.lit("page_visit").alias("event_type"),
        pl.lit(None).cast(pl.Float64).alias("sku"),
    ]).select(["client_id", "timestamp", "event_type", "sku"])
    return df


def sessionize(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sort by (client_id, timestamp), cut a new session on first event per user
    or whenever the gap to the previous event exceeds SESSION_TIMEOUT_MIN.
    session_id is a global monotonically-increasing int32.
    """
    timeout = pl.duration(minutes=SESSION_TIMEOUT_MIN)
    df = df.sort(["client_id", "timestamp"])
    df = df.with_columns(
        pl.col("timestamp").diff().over("client_id").alias("_diff")
    )
    df = df.with_columns(
        (pl.col("_diff").is_null() | (pl.col("_diff") > timeout)).alias("_new_sess")
    )
    df = df.with_columns(
        pl.col("_new_sess").cum_sum().cast(pl.Int32).alias("session_id")
    )
    return df.drop(["_diff", "_new_sess"])


def main():
    print("\nLoading item-bearing events...")
    atc = load_item_bearing("add_to_cart")
    rfc = load_item_bearing("remove_from_cart")
    buy = load_item_bearing("product_buy")

    print("\nLoading non-item events...")
    sq = load_search_query()
    pv = load_page_visit()

    print("\nMerging...")
    df = pl.concat([atc, rfc, buy, sq, pv], how="vertical")

    print(f"\nSessionizing (timeout = {SESSION_TIMEOUT_MIN} min)...")
    df = sessionize(df)

    n_events   = df.height
    n_users    = df["client_id"].n_unique()
    n_sessions = df["session_id"].n_unique()

    # Inter-session gap distribution
    session_starts = (
        df.group_by(["client_id", "session_id"])
        .agg(pl.col("timestamp").min().alias("session_start"))
        .sort(["client_id", "session_start"])
        .with_columns(
            pl.col("session_start").diff().over("client_id").alias("gap")
        )
        .filter(pl.col("gap").is_not_null())
    )
    gap_hours = (session_starts["gap"].dt.total_seconds() / 3600)

    print("\nInter-session gap distribution (hours):")
    print(f"n gaps          : {len(gap_hours):,}")
    print(f"mean            : {gap_hours.mean():.1f}h")
    print(f"median (p50)    : {gap_hours.quantile(0.50):.1f}h")
    print(f"p75             : {gap_hours.quantile(0.75):.1f}h")
    print(f"p95             : {gap_hours.quantile(0.95):.1f}h")
    print(f"p99             : {gap_hours.quantile(0.99):.1f}h")
    print(f"max             : {gap_hours.max():.1f}h")

    # Drop sessions that contain only page_visit events (no SKU signal)
    has_item_event = (
        df.filter(pl.col("event_type") != "page_visit")
        .select("session_id")
        .unique()
    )
    n_before = df["session_id"].n_unique()
    df = df.join(has_item_event, on="session_id", how="inner")
    n_after = df["session_id"].n_unique()
    print(f"\nFiltered page-visit-only sessions: {n_before:,} -> {n_after:,} ({n_before - n_after:,} dropped)")

    # Split on session start: train+val -> clean, test -> locked separate file
    val_cut = pd.Timestamp(VAL_CUTOFF).to_pydatetime()
    sess_starts = (
        df.group_by("session_id")
        .agg(pl.col("timestamp").min().alias("_sess_start"))
    )
    df = df.join(sess_starts, on="session_id")
    df_trainval = df.filter(pl.col("_sess_start") <  val_cut).drop("_sess_start")
    df_test     = df.filter(pl.col("_sess_start") >= val_cut).drop("_sess_start")

    print(f"\nSplit by VAL_CUTOFF ({VAL_CUTOFF}):")
    print(f"  train+val sessions : {df_trainval['session_id'].n_unique():>10,}  ({df_trainval.height:,} events)")
    print(f"  test sessions      : {df_test['session_id'].n_unique():>10,}  ({df_test.height:,} events)")

    print(f"\nSaving train+val -> {CLEAN_PARQUET}")
    df_trainval.write_parquet(CLEAN_PARQUET)
    print(f"Saving test      -> {TEST_PARQUET}  (locked, do not use for training or EDA)")
    df_test.write_parquet(TEST_PARQUET)
    print()
    print(f"Total events    : {n_events:>12,}")
    print(f"Unique users    : {n_users:>12,}")
    print(f"Sessions        : {n_sessions:>12,}")
    print(f"Saved clean     : {CLEAN_PARQUET}")
    print(f"Saved test      : {TEST_PARQUET}")


if __name__ == "__main__":
    main()
