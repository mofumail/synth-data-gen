import polars as pl
import numpy as np

# Load training data
df = pl.read_parquet("../pipeline/output/events_clean.parquet")

# Calculate counts
stats = df.group_by("event_type").count()
total = df.height

# Apply Log-Inverse-Frequency formula
# log(Total/Count)
stats = stats.with_columns(
    weight = (total / pl.col("count")).log()
)

# Print for YAML
print("svdpq_event_weights:")
for row in stats.iter_rows():
    etype, count, weight = row
    # Make 'remove_from_cart' negative as its a negative signal? I assume that since removing items carts
    # signifies that the user dislikes the product or is uninterested in it; it should be a negative signal.
     #if etype == "remove_from_cart":
       #  weight = -weight
    print(f"  {etype:20}: {weight:.4f}")