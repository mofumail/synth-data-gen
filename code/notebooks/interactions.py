import matplotlib.pyplot as plt
import numpy as np
import polars as pl

VOCAB_SIZE = 1_500_000

df = pl.read_parquet("../pipeline/output/events_clean.parquet")

# Cold-start users: share falling under each interaction threshold
user_interactions = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .group_by("client_id")
    .agg(pl.len().alias("n_interactions"))
)
total_users = user_interactions.height

for t in range(1, 20):
    cold = user_interactions.filter(pl.col("n_interactions") < t).height
    print(f"  threshold < {t:2d} : {cold:>8,} cold-start users  ({cold/total_users*100:.1f}%)")

# Cold SKUs (buy/add interactions only)
sku_interactions = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .filter(pl.col("sku").is_not_null())
    .group_by("sku")
    .agg(pl.len().alias("n_interactions"))
)
total_skus = sku_interactions.height

plot_thresholds = list(range(1, 20))
cold_counts = []
cold_pcts = []
for t in plot_thresholds:
    cold = sku_interactions.filter(pl.col("n_interactions") < t).height
    cold_pct = cold / total_skus * 100
    cold_counts.append(cold)
    cold_pcts.append(cold_pct)
    print(f"  threshold < {t:2d} : {cold:>8,} cold SKUs  ({cold_pct:.1f}%)")

fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = "tab:blue"
ax1.set_xlabel("Interaction Threshold (n_interactions < t)", fontweight="bold")
ax1.set_ylabel("Percentage of Cold SKUs (%)", color=color1, fontweight="bold")
ax1.plot(plot_thresholds, cold_pcts, marker="o", color=color1, linewidth=2)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(plot_thresholds)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Count of Cold SKUs", color=color2, fontweight="bold")
ax2.plot(plot_thresholds, cold_counts, marker="s", color=color2, linestyle=":", alpha=0.7)
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Cold SKUs vs. Interaction Threshold", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("cold_skus_threshold_plot.png", dpi=300)
plt.close()

# Cold SKUs that still appear as training targets
train_targets = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .filter(pl.col("sku").is_not_null())
    .select("sku")
    .unique()
)
cold_skus = sku_interactions.filter(pl.col("n_interactions") < 3).select("sku")
cold_but_targeted = cold_skus.join(train_targets, on="sku", how="inner")
print(f"Cold SKUs that appear as targets: {cold_but_targeted.height:,} / {cold_skus.height:,}")

# SKUs in vocab but with zero interactions
truly_unseen = VOCAB_SIZE - sku_interactions.height
print(f"Truly unseen SKUs (0 interactions): {truly_unseen:,}")

# Event-type distribution and IDF weights
stats = (
    df.group_by("event_type")
    .agg(pl.len().alias("count"))
    .with_columns(pct=(pl.col("count") / df.height * 100).round(2))
    .sort("count", descending=True)
)
print(stats)
for etype, count, pct in stats.iter_rows():
    weight = np.log(df.height / count)
    print(f"  {etype:20}: {weight:.4f}  ({pct}% of events)")

# Session-level event-type co-occurrence
sess = df.group_by("session_id").agg(pl.col("event_type").unique().alias("types"))
for a, b in [
    ("add_to_cart", "product_buy"),
    ("add_to_cart", "remove_from_cart"),
    ("remove_from_cart", "product_buy"),
]:
    both = sess.filter(pl.col("types").list.contains(a) & pl.col("types").list.contains(b)).height
    has_a = sess.filter(pl.col("types").list.contains(a)).height
    print(f"  {a} + {b}: {both:,} sessions  ({both/has_a*100:.1f}% of sessions with {a})")

# SKU interaction counts across all event types
sku_counts = (
    df.filter(pl.col("sku").is_not_null())
    .group_by("sku")
    .agg(pl.len().alias("n"))
)
total_skus_all = sku_counts.height
for t in [2, 3, 5, 10]:
    cold = sku_counts.filter(pl.col("n") < t).height
    print(f"  threshold < {t}: {cold:,} cold SKUs ({cold/total_skus_all*100:.1f}%)")

# What fraction of cold SKUs are actually buy/add targets?
buy_add_targets = (
    df.filter(pl.col("event_type").is_in(["product_buy", "add_to_cart"]))
    .filter(pl.col("sku").is_not_null())
    .select("sku")
    .unique()
)
for thresh in [2, 3]:
    cold = sku_counts.filter(pl.col("n") < thresh).select("sku")
    targeted = cold.join(buy_add_targets, on="sku", how="inner").height
    print(f"  threshold < {thresh}: {targeted:,}/{cold.height:,} cold SKUs are buy/add targets ({targeted/cold.height*100:.1f}%)")

# Remove -> buy signal within a session
removed = df.filter(pl.col("event_type") == "remove_from_cart").select(["session_id", "sku"])
bought = df.filter(pl.col("event_type") == "product_buy").select(["session_id", "sku"])
remove_then_buy = removed.join(bought, on=["session_id", "sku"], how="inner").height
print(f"  remove -> buy same SKU same session: {remove_then_buy:,} / {removed.height:,} removes ({remove_then_buy/removed.height*100:.1f}%)")
