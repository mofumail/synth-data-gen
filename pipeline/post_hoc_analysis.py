"""
post_hoc_analysis.py

Post-hoc analyses on the saved GRU4Rec models and synthetic session parquets.
No retraining or regeneration required.

Run from pipeline/:
    PYTHONPATH=. uv run python post_hoc_analysis.py

Analyses:
    1. OOV breakdown - HR@10 on OOV vs non-OOV test pairs per condition
    2. Markov fidelity - fidelity metrics for Markov sessions vs real
    3. Context length - HR@10 by context length bucket per condition
    4. Item coverage by popularity rank - which items does the transformer generate?
    5. OOV rate comparison across conditions (from eval logs, confirmed here)
"""

import sys
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from collections import defaultdict

from config import OUTPUT_DIR, GRU4REC_EMBED_DIM, GRU4REC_HIDDEN_DIM
from evaluation.downstream import _GRU4RecModel
from evaluation.reference import RealDataLoader, ReferenceStore
from evaluation.fidelity import FidelityEvaluator

MODELS_DIR  = OUTPUT_DIR / "models" / "gru4rec"
SYNTH_DIR   = OUTPUT_DIR / "synthetic"
REF_STORE   = OUTPUT_DIR / "reference_store.joblib"
SKU2IDX     = OUTPUT_DIR / "sku2idx.joblib"
K           = 10

CONDITIONS = {
    "TSTR-T": MODELS_DIR / "session_transformer_d64_l2-tstr_t-500000.pt",
    "TSTR-M": MODELS_DIR / "session_transformer_d64_l2-tstr_m-500000.pt",
    "TRTR":   MODELS_DIR / "session_transformer_d64_l2-trtr-500000.pt",
}

# Helpers

def load_gru4rec(path: Path):
    ckpt     = torch.load(path, map_location="cpu", weights_only=False)
    item2idx = ckpt["item2idx"]
    idx2sku  = {v: k for k, v in item2idx.items()}
    n_items  = len(item2idx)
    model    = _GRU4RecModel(n_items, GRU4REC_EMBED_DIM, GRU4REC_HIDDEN_DIM)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, item2idx, idx2sku


def build_test_pairs(sessions):
    pairs = []
    for session in sessions:
        item_evs = [e for e in session if e.get("sku") is not None]
        if len(item_evs) < 2:
            continue
        pairs.append((item_evs[:-1], int(item_evs[-1]["sku"])))
    return pairs


def predict_topk(model, item2idx, idx2sku, context, k=10):
    seq = [item2idx[ev["sku"]] for ev in context
           if ev.get("sku") is not None and ev["sku"] in item2idx]
    if not seq:
        return []
    x = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    last = logits[0, -1].clone()
    last[0] = float("-inf")
    topk = torch.topk(last, min(k, last.size(0) - 1)).indices.tolist()
    return [idx2sku[i] for i in topk if i in idx2sku]


def hr(preds, gt, k=10):
    return 1.0 if gt in preds[:k] else 0.0


def ndcg(preds, gt, k=10):
    for rank, item in enumerate(preds[:k], start=1):
        if item == gt:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def parquet_to_sessions(path: Path):
    df = pd.read_parquet(path)
    sessions = []
    for sid, grp in df.groupby("session_id", sort=True):
        grp = grp.sort_values("timestamp") if "timestamp" in grp.columns else grp
        sessions.append(grp.to_dict("records"))
    return sessions


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# Load shared data

print("Loading val split (100k sessions) ...")
real_data  = RealDataLoader.load(max_val_sessions=100_000)
test_pairs = build_test_pairs(real_data.val_split)
print(f"  {len(test_pairs):,} test pairs")

ref_store = ReferenceStore.load(REF_STORE)
sku2idx   = joblib.load(SKU2IDX)   # sku -> popularity rank (1 = most popular)

# Analysis 1 & 5: OOV breakdown per condition

section("1 & 5 - OOV rate + HR@10 breakdown (OOV vs non-OOV)")

oov_summary = {}
for cond, model_path in CONDITIONS.items():
    print(f"\n  [{cond}]")
    model, item2idx, idx2sku = load_gru4rec(model_path)

    oov_pairs    = [(ctx, gt) for ctx, gt in test_pairs if gt not in item2idx]
    non_oov_pairs = [(ctx, gt) for ctx, gt in test_pairs if gt in item2idx]

    oov_rate = len(oov_pairs) / len(test_pairs)
    print(f"    OOV: {len(oov_pairs):,}/{len(test_pairs):,} ({oov_rate:.1%})")

    # HR on non-OOV only
    hr_non_oov = np.mean([hr(predict_topk(model, item2idx, idx2sku, ctx, K), gt, K)
                          for ctx, gt in non_oov_pairs]) if non_oov_pairs else 0.0

    # HR overall (OOV pairs contribute 0)
    hr_overall = hr_non_oov * (1 - oov_rate)

    print(f"    HR@{K} overall  : {hr_overall:.4f}")
    print(f"    HR@{K} non-OOV  : {hr_non_oov:.4f}  (n={len(non_oov_pairs):,})")
    print(f"    HR@{K} OOV      : 0.0000  (guaranteed, n={len(oov_pairs):,})")

    oov_summary[cond] = dict(oov_rate=oov_rate, hr_overall=hr_overall, hr_non_oov=hr_non_oov)

print("\n  Summary table:")
print(f"  {'Condition':<10} {'OOV%':>8} {'HR@10 overall':>15} {'HR@10 non-OOV':>15}")
for cond, d in oov_summary.items():
    print(f"  {cond:<10} {d['oov_rate']:>7.1%} {d['hr_overall']:>15.4f} {d['hr_non_oov']:>15.4f}")

# Analysis 2: Markov fidelity

section("2 - Markov fidelity vs real (train reference)")

markov_path = next(SYNTH_DIR.glob("markov-*.parquet"), None)
if markov_path:
    print(f"  Loading {markov_path.name} ...")
    markov_sessions = parquet_to_sessions(markov_path)
    print(f"  {len(markov_sessions):,} sessions loaded")

    fid_ev = FidelityEvaluator()
    markov_fidelity = fid_ev.evaluate(None, markov_sessions, ref_store)

    print(f"\n  Markov fidelity vs train:")
    print(f"    JSD(action)        = {markov_fidelity.jsd_action:.4f}")
    print(f"    KS(session_len)    = {markov_fidelity.ks_session_length:.4f}")
    print(f"    KS(temporal_delta) = {markov_fidelity.ks_temporal_delta:.4f}")
    print(f"    L1(action bigrams) = {markov_fidelity.l1_action_bigrams:.4f}")
    print(f"    diversity          = {markov_fidelity.sample_diversity:.4f}")
    print(f"    conv_rate_delta    = {markov_fidelity.conversion_rate_delta:.4f}")
    print(f"    cart_abandon_delta = {markov_fidelity.cart_abandonment_delta:.4f}")
    print(f"    JSD(popularity)    = {markov_fidelity.bias.popularity_jsd:.4f}")
    print(f"    item_coverage      = {markov_fidelity.bias.item_coverage:.4f}")
else:
    print("  Markov parquet not found - skipping")

# Analysis 3: HR@10 by context length bucket

section("3 - HR@10 by context length (1 item | 2-3 | 4+ items in context)")

buckets = {"1": [], "2-3": [], "4+": []}

def bucket_of(ctx):
    n = len([e for e in ctx if e.get("sku") is not None])
    if n == 1:   return "1"
    if n <= 3:   return "2-3"
    return "4+"

print(f"\n  {'Condition':<10} {'ctx=1':>10} {'ctx=2-3':>10} {'ctx=4+':>10} {'n(1)':>8} {'n(2-3)':>8} {'n(4+)':>8}")
for cond, model_path in CONDITIONS.items():
    model, item2idx, idx2sku = load_gru4rec(model_path)
    results = defaultdict(list)
    for ctx, gt in test_pairs:
        b    = bucket_of(ctx)
        pred = predict_topk(model, item2idx, idx2sku, ctx, K)
        results[b].append(hr(pred, gt, K))

    r = {b: (np.mean(v) if v else 0.0, len(v)) for b, v in results.items()}
    print(f"  {cond:<10} "
          f"{r.get('1',  (0,0))[0]:>10.4f} "
          f"{r.get('2-3',(0,0))[0]:>10.4f} "
          f"{r.get('4+', (0,0))[0]:>10.4f} "
          f"{r.get('1',  (0,0))[1]:>8,} "
          f"{r.get('2-3',(0,0))[1]:>8,} "
          f"{r.get('4+', (0,0))[1]:>8,}")

# Analysis 4: Item coverage by popularity rank

section("4 - Transformer item coverage by popularity rank")

synth_path = next(SYNTH_DIR.glob("session_transformer_d64_l2-*.parquet"), None)
if synth_path:
    synth_df    = pd.read_parquet(synth_path)
    synth_skus  = set(synth_df["sku"].dropna().astype(int).unique())
    total_items = len(sku2idx)

    buckets_cov = [
        ("top 1k",       1,       1_000),
        ("1k – 10k",     1_001,   10_000),
        ("10k – 50k",    10_001,  50_000),
        ("50k – 100k",   50_001,  100_000),
        ("100k – 630k",  100_001, total_items),
    ]

    print(f"\n  Synthetic sessions: {synth_path.name}")
    print(f"  Unique items in synthetic : {len(synth_skus):,}")
    print(f"  Total items in real vocab : {total_items:,}")
    print()
    print(f"  {'Rank bucket':<20} {'real items':>12} {'synth covered':>14} {'coverage':>10}")
    for label, lo, hi in buckets_cov:
        real_in_bucket  = {sku for sku, idx in sku2idx.items() if lo <= idx <= hi}
        synth_in_bucket = synth_skus & real_in_bucket
        pct = len(synth_in_bucket) / len(real_in_bucket) if real_in_bucket else 0.0
        print(f"  {label:<20} {len(real_in_bucket):>12,} {len(synth_in_bucket):>14,} {pct:>9.1%}")
else:
    print("  Transformer parquet not found - skipping")

print("\nDone.")
