"""
The `name` column in product_properties contains 16 × uint8 values —
an LLM embedding that was quantised to 128 bits. This script tests whether
those bytes retain any semantic structure
Tests performed:
  1. Value distribution per byte position (uniform ≈ no structure)
  2. Intra- vs inter-category cosine similarity (should differ if semantic)
  3. Nearest-neighbour category agreement (should be above chance if semantic)
  4. t-SNE / PCA visual inspection data (optional export)
  5. Mutual information between name bytes and category
"""

import re
import numpy as np
import polars as pl
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import mutual_info_score
from collections import Counter


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "DATA"
PROPS_PATH = DATA_DIR / "product_properties.parquet"
SEED = 42
N_SAMPLE = None  # subsample for expensive pairwise ops
N_CATEGORIES_SAMPLE = None  # categories to sample for intra/inter comparison


def parse_name_bytes(s: str) -> list[int]:
    """Parse '[193 102 221 ...]' → list of ints."""
    return [int(x) for x in re.findall(r"\d+", s)][:16]


def load_data():
    """Load product_properties and return name matrix + categories."""
    print(f"Loading {PROPS_PATH} ...")
    pp = pl.read_parquet(PROPS_PATH)
    print(f"  {pp.height:,} items")

    name_raw = [parse_name_bytes(s) for s in pp["name"].to_list()]
    name_arr = np.array(
        [nb[:16] + [0] * max(0, 16 - len(nb)) for nb in name_raw],
        dtype=np.float32,
    )  # [N, 16], raw uint8 values as float

    categories = pp["category"].to_numpy().astype(np.int64)
    return name_arr, categories



def test_byte_distribution(name_arr: np.ndarray):
    """Check if byte values are roughly uniform (0–255) per position."""
    print("TEST 1: Byte value distribution per position")
    print("If embeddings are meaningful, we'd expect non-uniform distributions.")
    print("If quantisation destroyed structure, values should be near-uniform.\n")

    for pos in range(16):
        vals = name_arr[:, pos].astype(int)
        unique = len(np.unique(vals))
        counts = np.array(list(Counter(vals).values()), dtype=np.float64)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(256)  # 8 bits = uniform over 256 values
        print(
            f"  Byte {pos:2d}: {unique:3d} unique values | "
            f"entropy={entropy:.3f} / {max_entropy:.3f} "
            f"({entropy / max_entropy * 100:.1f}% of max)"
        )

    # Overall
    all_vals = name_arr.flatten().astype(int)
    overall_unique = len(np.unique(all_vals))
    counts = np.array(list(Counter(all_vals).values()), dtype=np.float64)
    probs = counts / counts.sum()
    overall_entropy = -np.sum(probs * np.log2(probs))
    print(f"\n  Overall: {overall_unique} unique values across all positions")
    print(f"  Overall entropy: {overall_entropy:.3f} / 8.000 ({overall_entropy / 8 * 100:.1f}%)")



def test_intra_inter_similarity(name_arr: np.ndarray, categories: np.ndarray):
    """Compare cosine similarity within vs between categories."""
    print("TEST 2: Intra-category vs inter-category cosine similarity")
    print("If semantic, intra-category similarity >> inter-category similarity.\n")

    rng = np.random.default_rng(SEED)

    # Pick categories with enough items
    cat_ids, cat_counts = np.unique(categories, return_counts=True)
    valid_cats = cat_ids[cat_counts >= 10]
    sampled_cats = rng.choice(valid_cats, size=min(N_CATEGORIES_SAMPLE, len(valid_cats)), replace=False)

    intra_sims = []
    inter_sims = []

    for cat in sampled_cats:
        mask = categories == cat
        cat_vecs = name_arr[mask]

        # Sample pairs for intra-category
        if len(cat_vecs) < 2:
            continue
        n_pairs = min(50, len(cat_vecs))
        idx = rng.choice(len(cat_vecs), size=n_pairs, replace=False)
        vecs = cat_vecs[idx]

        # Intra-category: pairwise cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs_normed = vecs / norms
        sim_matrix = vecs_normed @ vecs_normed.T
        triu_idx = np.triu_indices(len(vecs_normed), k=1)
        if len(triu_idx[0]) > 0:
            intra_sims.extend(sim_matrix[triu_idx].tolist())

        # Inter-category: compare against random items from other categories
        other_mask = ~mask
        other_idx = rng.choice(np.where(other_mask)[0], size=n_pairs, replace=False)
        other_vecs = name_arr[other_idx]
        other_norms = np.linalg.norm(other_vecs, axis=1, keepdims=True)
        other_norms = np.where(other_norms == 0, 1, other_norms)
        other_normed = other_vecs / other_norms

        for i in range(len(vecs_normed)):
            for j in range(len(other_normed)):
                inter_sims.append(np.dot(vecs_normed[i], other_normed[j]))

    intra_mean = np.mean(intra_sims)
    inter_mean = np.mean(inter_sims)
    intra_std = np.std(intra_sims)
    inter_std = np.std(inter_sims)

    print(f"Intra-category cosine similarity: {intra_mean:.4f} ± {intra_std:.4f}  (n={len(intra_sims):,})")
    print(f"nter-category cosine similarity: {inter_mean:.4f} ± {inter_std:.4f}  (n={len(inter_sims):,})")
    print(f"Difference (intra - inter):       {intra_mean - inter_mean:.4f}")



def test_nn_category_agreement(name_arr: np.ndarray, categories: np.ndarray):
    """For each item, check if its nearest neighbour shares the same category."""
    print("TEST 3: Nearest-neighbour category agreement")
    print("If semantic, NN should share category more often than random chance.\n")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(name_arr), size=min(N_SAMPLE, len(name_arr)), replace=False)
    vecs = name_arr[idx]
    cats = categories[idx]

    # Compute pairwise L2 distances in batches to avoid OOM
    batch_size = 5000
    n = len(vecs)
    same_cat_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = cdist(vecs[start:end], vecs, metric="euclidean")
        # Set self-distance to inf
        for i in range(end - start):
            dists[i, start + i] = np.inf
        nn_idx = np.argmin(dists, axis=1)
        same_cat_count += np.sum(cats[start:end] == cats[nn_idx])

    nn_agreement = same_cat_count / n
    n_cats = len(np.unique(cats))
    random_chance = 1.0 / n_cats

    # Also compute a category-frequency-weighted random baseline
    cat_counts = Counter(cats)
    weighted_chance = sum((c / n) ** 2 for c in cat_counts.values())

    print(f"NN same-category rate:      {nn_agreement:.4f} ({nn_agreement * 100:.2f}%)")
    print(f"Random chance (uniform):    {random_chance:.4f} ({random_chance * 100:.2f}%)")
    print(f"Random chance (freq-weighted): {weighted_chance:.4f} ({weighted_chance * 100:.2f}%)")
    print(f"Lift over freq-weighted:    {nn_agreement / weighted_chance:.2f}x")


def test_mutual_information(name_arr: np.ndarray, categories: np.ndarray):
    """Compute MI between each byte position and the category label."""
    print("TEST 4: Mutual information (name bytes → category)")
    print("High MI = byte position is predictive of category.\n")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(name_arr), size=min(N_SAMPLE, len(name_arr)), replace=False)
    cats = categories[idx]

    mis = []
    for pos in range(16):
        byte_vals = name_arr[idx, pos].astype(int)
        mi = mutual_info_score(byte_vals, cats)
        mis.append(mi)
        print(f"  Byte {pos:2d}: MI = {mi:.4f} nats")

    # Baseline: MI between random bytes and category
    random_bytes = rng.integers(0, 256, size=len(cats))
    random_mi = mutual_info_score(random_bytes, cats)
    print(f"Random baseline MI: {random_mi:.4f} nats")
    print(f"Mean name byte MI: {np.mean(mis):.4f} nats")

def test_pca_variance(name_arr: np.ndarray):
    """Check if PCA reveals any low-dimensional structure."""
    print("TEST 5: PCA variance explained")
    print("If structured, first few PCs should explain disproportionate variance.")
    print("If random, variance should be ~uniform across components.\n")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(name_arr), size=min(N_SAMPLE, len(name_arr)), replace=False)
    vecs = name_arr[idx]

    # Center
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    for i in range(16):
        print(f"  PC {i + 1:2d}: {explained[i] * 100:5.2f}% (cumulative: {cumulative[i] * 100:5.1f}%)")

    # Compare to what uniform random data would look like
    # For 16 uniform dimensions, each PC explains ~6.25%
    print(f"Expected if uniform random: ~{100 / 16:.2f}% per component")
    print(f"Top-3 PCs explain: {cumulative[2] * 100:.1f}% (vs {3 * 100 / 16:.1f}% if random)")



def main():
    name_arr, categories = load_data()

    test_byte_distribution(name_arr)
    test_intra_inter_similarity(name_arr, categories)
    test_nn_category_agreement(name_arr, categories)
    test_mutual_information(name_arr, categories)
    test_pca_variance(name_arr)

if __name__ == "__main__":
    main()
