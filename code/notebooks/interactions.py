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