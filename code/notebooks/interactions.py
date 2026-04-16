import polars as pl


df = pl.read_parquet("../pipeline/output/events_clean.parquet")

# Count interactions per user
user_interactions = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .group_by("client_id")
    .agg(pl.len().alias("n_interactions"))
)

total_users = user_interactions.height

# For each candidate threshold, what % of users fall into cold-start?
thresholds = range(1, 20)
for t in thresholds:
    cold = user_interactions.filter(pl.col("n_interactions") < t).height
    print(f"  threshold < {t:2d} : {cold:>8,} cold-start users  ({cold/total_users*100:.1f}%)")

sku_interactions = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .filter(pl.col("sku").is_not_null())
    .group_by("sku")
    .agg(pl.len().alias("n_interactions"))
)

total_skus = sku_interactions.height
for t in range(1, 15):
    cold = sku_interactions.filter(pl.col("n_interactions") < t).height
    print(f"  threshold < {t:2d} : {cold:>8,} cold SKUs  ({cold/total_skus*100:.1f}%)")


# SKUs that are cold but still appear as training targets
train_targets = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .filter(pl.col("sku").is_not_null())
    .select("sku")
    .unique()
)

cold_skus = sku_interactions.filter(pl.col("n_interactions") < 3).select("sku")

cold_but_targeted = cold_skus.join(train_targets, on="sku", how="inner")
print(f"Cold SKUs that appear as targets: {cold_but_targeted.height:,} / {cold_skus.height:,}")


# How many SKUs have 0 interactions (truly unseen)?
truly_unseen = (1500000 - sku_interactions.height)  # SKUs in vocab but not in interaction data
print(f"Truly unseen SKUs (0 interactions): {truly_unseen:,}")


import polars as pl
import numpy as np

df = pl.read_parquet("../pipeline/output/events_clean.parquet")

# 1. Raw event distribution
print("=== Event Distribution ===")
stats = (
    df.group_by("event_type")
    .agg(pl.len().alias("count"))
    .with_columns(pct=(pl.col("count") / df.height * 100).round(2))
    .sort("count", descending=True)
)
print(stats)

# 2. IDF weights (what your formula actually produces)
print("\n=== IDF Weights (log(N/count)) ===")
for row in stats.iter_rows():
    etype, count, pct = row
    weight = np.log(df.height / count)
    print(f"  {etype:20}: {weight:.4f}  ({pct}% of events)")

# 3. Co-occurrence: do buys co-occur with adds in same session?
print("\n=== Session Co-occurrence ===")
sess = df.group_by("session_id").agg(pl.col("event_type").unique().alias("types"))
total_sess = sess.height
for a, b in [("add_to_cart", "product_buy"), ("add_to_cart", "remove_from_cart"), ("remove_from_cart", "product_buy")]:
    both = sess.filter(
        pl.col("types").list.contains(a) & pl.col("types").list.contains(b)
    ).height
    has_a = sess.filter(pl.col("types").list.contains(a)).height
    print(f"  {a} + {b}: {both:,} sessions  ({both/has_a*100:.1f}% of sessions with {a})")

# 4. SKU interaction distribution (for svdpq_min_interactions)
print("\n=== SKU Interaction Counts ===")
sku_counts = (
    df.filter(pl.col("sku").is_not_null())
    .group_by("sku")
    .agg(pl.len().alias("n"))
)
total_skus = sku_counts.height
for t in [2, 3, 5, 10]:
    cold = sku_counts.filter(pl.col("n") < t).height
    print(f"  threshold < {t}: {cold:,} cold SKUs ({cold/total_skus*100:.1f}%)")

# 5. For cold SKUs (< 3), what fraction are buy/add targets specifically?
print("\n=== Cold SKU Target Quality ===")
for thresh in [2, 3]:
    cold_skus = sku_counts.filter(pl.col("n") < thresh).select("sku")
    buy_add = (
        df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
        .filter(pl.col("sku").is_not_null())
        .select("sku").unique()
    )
    targeted = cold_skus.join(buy_add, on="sku", how="inner").height
    print(f"  threshold < {thresh}: {targeted:,}/{cold_skus.height:,} cold SKUs are buy/add targets ({targeted/cold_skus.height*100:.1f}%)")

# 6. Remove signal: does remove precede buy of same item?
print("\n=== Remove -> Buy (same SKU, same session) ===")
removed = df.filter(pl.col("event_type") == "remove_from_cart").select(["session_id", "sku"])
bought  = df.filter(pl.col("event_type") == "product_buy").select(["session_id", "sku"])
remove_then_buy = removed.join(bought, on=["session_id", "sku"], how="inner").height
print(f"  remove -> buy same SKU same session: {remove_then_buy:,} / {removed.height:,} removes ({remove_then_buy/removed.height*100:.1f}%)")