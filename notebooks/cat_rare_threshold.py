import polars as pl

props = pl.read_parquet("../DATA/product_properties.parquet", columns=["sku", "category"])

cat_sizes = (
    props.group_by("category")
    .agg(pl.len().alias("n_skus"))
    .sort("n_skus", descending=True)
)

total_cats = cat_sizes.height
total_skus = props.height

for t in [1, 2, 3, 5, 10, 20, 50]:
    rare = cat_sizes.filter(pl.col("n_skus") <= t)
    rare_cats = rare.height
    rare_skus = rare["n_skus"].sum()
    print(f"  threshold <= {t:2d}: {rare_cats:>6,} rare cats ({rare_cats/total_cats*100:.1f}%)  "
          f"covering {rare_skus:>8,} SKUs ({rare_skus/total_skus*100:.1f}% of catalog)")