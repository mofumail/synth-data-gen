"""
Popularity-bias metrics for recommendation baskets.

Implements the four metrics proposed in Braun, Bhaumik & Dey (2023),
"Metrics for popularity bias in dynamic recommender systems"
(arXiv:2310.08455), adapted to the basket parquet produced by
generate_baskets.py:

    1. Within-group Gini coefficient   (paper sec. 3.1, eq. 1-2)
    2. Dynamic-dGAP                    (paper sec. 3.2, eq. 3-4)
    3. Between-group GAP               (paper sec. 3.3, eq. 6-7)
    4. Group cosine similarity         (paper sec. 3.3)

The paper measures these across *sensitive* user groups (males/females).
The Synerise data carries no demographic attributes, so groups are
derived from each user's training profile instead:

    profile_pop : quantile split on the mean popularity score of the
                  items in the user's train profile (niche-focused vs
                  mainstream-focused users, the grouping used by
                  Abdollahpouri et al., 2019 for dGAP).
    activity    : quantile split on train profile size (number of
                  distinct items interacted with).

The paper's dynamic axis is a simulated feedback loop; here each basket
file is a single feedback step of one model iteration, and the dynamic
reading comes from comparing metric values across model iterations.

Item popularity scores (phi, eq. 1) and user profiles are computed from
the train split of events_clean.parquet using the same session-start
cutoff as evaluation.baskets, so basket users join exactly onto their
generation-time history.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import polars as pl

from config import CLEAN_PARQUET, TRAIN_CUTOFF


# Positive item interactions only: matches the rec_action choices offered
# by generate_baskets.py (remove_from_cart is excluded as a negative signal).
POSITIVE_ITEM_EVENTS: tuple[str, ...] = ("product_buy", "add_to_cart")

GROUP_STRATEGIES = ("profile_pop", "activity")

_GROUP_LABELS_2 = {
    "profile_pop": ("niche", "mainstream"),
    "activity": ("low_activity", "high_activity"),
}


# --------------------------------------------------------------------------
# Train-side inputs: interactions, popularity scores, profiles, groups
# --------------------------------------------------------------------------

def read_train_interactions(
    events_path: Path = CLEAN_PARQUET,
    train_cutoff=TRAIN_CUTOFF,
    item_events: Iterable[str] = POSITIVE_ITEM_EVENTS,
) -> pd.DataFrame:
    """
    Distinct (user_id, item_id) interaction pairs from the train split.

    Train membership is decided on session start, mirroring
    evaluation.baskets._read_train_events, so the profile seen here is the
    same history the generator was conditioned on.
    """
    columns = ["client_id", "timestamp", "event_type", "session_id", "sku"]
    df = pl.read_parquet(str(events_path), columns=columns)
    session_starts = df.group_by(["client_id", "session_id"]).agg(
        pl.col("timestamp").min().alias("session_start")
    )
    return (
        df.join(session_starts, on=["client_id", "session_id"])
        .filter(pl.col("session_start") < pd.Timestamp(train_cutoff))
        .filter(
            pl.col("event_type").is_in(list(item_events))
            & pl.col("sku").is_not_null()
        )
        .select(
            pl.col("client_id").alias("user_id"),
            pl.col("sku").cast(pl.Int64).alias("item_id"),
        )
        .unique()
        .to_pandas()
    )


def item_popularity(interactions: pd.DataFrame) -> pd.Series:
    """
    Popularity score phi_i = N_i / N_U (eq. 1): fraction of train users
    that interacted with item i. Indexed by item_id.
    """
    n_users = interactions["user_id"].nunique()
    return interactions.groupby("item_id")["user_id"].nunique() / n_users


def profile_stats(interactions: pd.DataFrame, phi: pd.Series) -> pd.DataFrame:
    """
    Per-user profile summary: profile size and mean popularity of profile
    items (the per-user inner term of GAP, eq. 3). Indexed by user_id.
    """
    df = interactions.assign(phi=interactions["item_id"].map(phi))
    return df.groupby("user_id").agg(
        profile_size=("item_id", "size"),
        profile_mean_pop=("phi", "mean"),
    )


def assign_groups(
    stats: pd.DataFrame,
    strategy: str = "profile_pop",
    n_groups: int = 2,
) -> pd.Series:
    """
    Quantile-split users into n_groups on the chosen profile statistic.
    Returns a user_id-indexed Series of group labels. Group boundaries
    depend only on train data, so labels are identical across the basket
    files of different model iterations.
    """
    if strategy not in GROUP_STRATEGIES:
        raise ValueError(f"strategy must be one of {GROUP_STRATEGIES}, got {strategy!r}")
    col = "profile_mean_pop" if strategy == "profile_pop" else "profile_size"

    if n_groups == 2 and strategy in _GROUP_LABELS_2:
        labels = list(_GROUP_LABELS_2[strategy])
    else:
        labels = [f"{strategy}_q{i + 1}" for i in range(n_groups)]

    # rank() breaks ties so qcut always yields the requested bin count even
    # on heavily tied statistics such as small integer profile sizes.
    ranked = stats[col].rank(method="first")
    return pd.qcut(ranked, q=n_groups, labels=labels).astype(str)


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (eq. 2)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * x).sum() / (n * total))


def gap(profile_items: pd.DataFrame, phi: pd.Series) -> float:
    """
    Group average popularity (eq. 3): mean over the group's users of the
    mean popularity score of the items in their list. profile_items has
    columns (user_id, item_id); items unseen in train score phi = 0.
    """
    if profile_items.empty:
        return float("nan")
    scores = profile_items["item_id"].map(phi).fillna(0.0)
    return float(scores.groupby(profile_items["user_id"]).mean().mean())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _group_popularity_counts(pairs: pd.DataFrame) -> pd.Series:
    """Distinct-user interaction counts per item (N_i within a group)."""
    return pairs.groupby("item_id")["user_id"].nunique()


# --------------------------------------------------------------------------
# Basket-level evaluation
# --------------------------------------------------------------------------

def load_basket_items(path: Path) -> pd.DataFrame:
    """
    Read a basket parquet into (user_id, item_id) recommendation pairs,
    dropping rank=-1 empty-basket sentinel rows. Returns the pairs plus the
    file-level metadata needed for reporting.
    """
    df = pd.read_parquet(path)
    valid = df[df["rank"] >= 0]
    pairs = (
        valid[["user_id", "item_id"]]
        .astype({"item_id": "int64"})
        .reset_index(drop=True)
    )
    meta = {
        "basket_file": str(path),
        "algorithm": str(df["algorithm"].iloc[0]) if len(df) else "unknown",
        "mode": str(df["mode"].iloc[0]) if len(df) else "unknown",
        "n_users": int(df["user_id"].nunique()),
        "n_users_with_basket": int(pairs["user_id"].nunique()),
        "mean_basket_size": float(pairs.groupby("user_id").size().mean())
        if len(pairs) else 0.0,
    }
    return pairs, meta


def evaluate_baskets(
    rec_pairs: pd.DataFrame,
    interactions: pd.DataFrame,
    phi: pd.Series,
    groups: pd.Series,
) -> dict:
    """
    Compute the four popularity-bias metrics for one basket file.

    rec_pairs    : (user_id, item_id) recommendations from load_basket_items
    interactions : distinct train (user_id, item_id) pairs
    phi          : item popularity scores from item_popularity()
    groups       : user_id-indexed group labels from assign_groups()

    Returns {"groups": {label: {...}}, "pairs": {"g|h": {...}},
             "n_users_evaluated": int, "n_users_unmatched": int}
    """
    known = rec_pairs["user_id"].isin(groups.index)
    n_unmatched = int(rec_pairs.loc[~known, "user_id"].nunique())
    rec_pairs = rec_pairs[known]

    rec_users = rec_pairs["user_id"].unique()
    user_group = groups.loc[rec_users]
    n_catalog = phi.size

    group_results: dict[str, dict] = {}
    group_vectors: dict[str, pd.Series] = {}

    for g in sorted(user_group.unique()):
        g_users = user_group[user_group == g].index
        g_recs = rec_pairs[rec_pairs["user_id"].isin(g_users)]
        g_profiles = interactions[interactions["user_id"].isin(g_users)]
        appended = pd.concat([g_profiles, g_recs], ignore_index=True)

        gap_p = gap(g_profiles, phi)
        gap_r = gap(g_recs, phi)
        delta_gap = (gap_r - gap_p) / gap_p if gap_p else float("nan")
        # eq. 6; phi values are << 1 in this data so the ratio sits near 1,
        # which is exactly the paper's "fair" reference point.
        delta_gap_revised = (1.0 - gap_r) / (1.0 - gap_p)

        # Within-group Gini over the popularity distribution of the group
        # dataset (profiles), the recommendations alone, and the dataset
        # with recommendations appended -- the paper's one-iteration update.
        gini_profiles = gini(_group_popularity_counts(g_profiles).to_numpy())
        rec_counts = g_recs.groupby("item_id").size()
        gini_recs = gini(rec_counts.to_numpy())
        gini_appended = gini(_group_popularity_counts(appended).to_numpy())

        group_results[g] = {
            "n_users": int(len(g_users)),
            "gap_p": gap_p,
            "gap_r": gap_r,
            "delta_gap": delta_gap,
            "delta_gap_revised": delta_gap_revised,
            "gini_profiles": gini_profiles,
            "gini_recs": gini_recs,
            "gini_profiles_plus_recs": gini_appended,
            # Catalog share reached by the recommendations: reads together
            # with gini_recs, since a collapsed generator can show a low
            # rec-Gini over the handful of items it still recommends.
            "item_coverage_recs": float(rec_counts.size / n_catalog) if n_catalog else float("nan"),
        }
        # Paper sec 3.3: frequency vector normalised by group size.
        group_vectors[g] = rec_counts / len(g_users)

    pair_results: dict[str, dict] = {}
    for g, h in combinations(sorted(group_results), 2):
        dg = group_results[g]["delta_gap_revised"]
        dh = group_results[h]["delta_gap_revised"]
        mean_d = (dg + dh) / 2.0
        between = abs(dg - dh) / mean_d if mean_d else float("nan")

        # Align the two frequency vectors on the union of recommended items;
        # items outside the union contribute zero to both vectors.
        union = group_vectors[g].index.union(group_vectors[h].index)
        vg = group_vectors[g].reindex(union, fill_value=0.0).to_numpy()
        vh = group_vectors[h].reindex(union, fill_value=0.0).to_numpy()

        pair_results[f"{g}|{h}"] = {
            "between_group_gap": between,
            "cosine_similarity": cosine_similarity(vg, vh),
        }

    return {
        "groups": group_results,
        "pairs": pair_results,
        "n_users_evaluated": int(len(rec_users)),
        "n_users_unmatched": n_unmatched,
    }


# --------------------------------------------------------------------------
# Tabular export
# --------------------------------------------------------------------------

def results_to_long_df(results_by_label: dict[str, dict]) -> pd.DataFrame:
    """
    Flatten {label: {meta..., metrics: evaluate_baskets() output}} into a
    long-form DataFrame (label, basket_file, group, metric, value) ready for
    matplotlib / seaborn.
    """
    rows: list[dict] = []
    for label, entry in results_by_label.items():
        base = {"label": label, "basket_file": entry["basket_file"]}
        metrics = entry["metrics"]
        for g, vals in metrics["groups"].items():
            for metric, value in vals.items():
                rows.append({**base, "group": g, "metric": metric, "value": value})
        for pair, vals in metrics["pairs"].items():
            for metric, value in vals.items():
                rows.append({**base, "group": pair, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["label", "basket_file", "group", "metric", "value"])
