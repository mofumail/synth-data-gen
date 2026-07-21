"""
plot_popularity_bias.py - Thesis figures for the popularity-bias evaluation

Reads the long-form CSVs written by evaluate_popularity_bias.py (one per
basket mode, labels like "iter1-s42" / "real-s42") and renders two figures
next to thesis.tex:

    popbias_dgap.{pdf,png}      dGAP per user group, topk vs rollout bars,
                                real-continuation reference band
    popbias_diversity.{pdf,png} Gini(recs), item coverage, cosine similarity,
                                faceted by mode (size-sensitive metrics are
                                only comparable within a mode; the real
                                reference is drawn on the rollout row, whose
                                basket sizes it matches)

Five-seed means; error bars / bands are one seed std.

Usage:
    PYTHONPATH=. uv run python plot_popularity_bias.py \
        --topk output/popularity_bias/topk-popularity_bias-<stamp>.csv \
        --rollout output/popularity_bias/rollout-popularity_bias-<stamp>.csv \
        --real output/popularity_bias/real-popularity_bias-<stamp>.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


THESIS_DIR = Path(__file__).parent          # thesis.tex lives next to this script
ITERS = ["iter1", "iter2", "iter3"]
ITER_NAMES = {"iter1": "Iter 1\n(flat)", "iter2": "Iter 2\n(hier)", "iter3": "Iter 3\n(svdpq)"}
GROUPS = ["mainstream", "niche"]
PAIR = "mainstream|niche"
MODE_NAMES = {"topk": "top-$k$ head", "rollout": "sampled sessions"}
REAL_COLOR = "#444444"


def load(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = source
    df["model"] = df["label"].str.rsplit("-s", n=1).str[0]
    df["seed"] = df["label"].str.rsplit("-s", n=1).str[1]
    return df


def agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds per (source, model, group, metric)."""
    return (
        df.groupby(["source", "model", "group", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )


def _bars(ax, stats, metric, group, sources, width=0.35):
    """Grouped bars for one metric+group across iterations and sources."""
    x = np.arange(len(ITERS))
    palette = sns.color_palette("colorblind")
    colors = {"topk": palette[0], "rollout": palette[2]}
    for i, src in enumerate(sources):
        sub = stats[(stats["source"] == src) & (stats["metric"] == metric)
                    & (stats["group"] == group)].set_index("model")
        means = [sub["mean"].get(it, np.nan) for it in ITERS]
        stds = [sub["std"].get(it, 0.0) for it in ITERS]
        offset = (i - (len(sources) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=colors[src], label=MODE_NAMES[src],
               error_kw={"lw": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels([ITER_NAMES[it] for it in ITERS])
    ax.axhline(0, color="black", lw=0.8)


def _real_band(ax, stats, metric, group, label="real continuation"):
    sub = stats[(stats["source"] == "real") & (stats["metric"] == metric)
                & (stats["group"] == group)]
    if sub.empty:
        return
    mean, std = float(sub["mean"].iloc[0]), float(sub["std"].iloc[0])
    ax.axhline(mean, color=REAL_COLOR, ls="--", lw=1.4, label=label, zorder=1)
    ax.axhspan(mean - std, mean + std, color=REAL_COLOR, alpha=0.12, zorder=0)


def fig_dgap(stats, out_stem: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, group in zip(axes, GROUPS):
        _bars(ax, stats, "delta_gap", group, ["topk", "rollout"])
        _real_band(ax, stats, "delta_gap", group)
        ax.set_title(f"{group} users")
        ax.set_ylabel(r"$\Delta$GAP" if group == GROUPS[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.13))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_diversity(stats, out_stem: Path):
    metrics = [
        ("gini_recs", "Gini (recommendations)", GROUPS),
        ("item_coverage_recs", "Item coverage", GROUPS),
        ("cosine_similarity", "Cosine niche$\\leftrightarrow$mainstream", [PAIR]),
    ]
    palette = sns.color_palette("colorblind")
    group_colors = {"mainstream": palette[1], "niche": palette[4], PAIR: palette[3]}

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), constrained_layout=True)
    for row, src in enumerate(["topk", "rollout"]):
        for col, (metric, title, groups) in enumerate(metrics):
            ax = axes[row, col]
            x = np.arange(len(ITERS))
            width = 0.35 if len(groups) > 1 else 0.5
            for i, group in enumerate(groups):
                sub = stats[(stats["source"] == src) & (stats["metric"] == metric)
                            & (stats["group"] == group)].set_index("model")
                means = [sub["mean"].get(it, np.nan) for it in ITERS]
                stds = [sub["std"].get(it, 0.0) for it in ITERS]
                offset = (i - (len(groups) - 1) / 2) * width
                ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                       color=group_colors[group],
                       label=group if row == 0 else None,
                       error_kw={"lw": 1.0})
                if src == "rollout":   # real baskets match rollout sizes only
                    _real_band(ax, stats, metric, group,
                               label="real continuation" if (col == 0 and i == 0) else None)
            ax.set_xticks(x)
            ax.set_xticklabels([ITER_NAMES[it] for it in ITERS], fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(MODE_NAMES[src], fontsize=10)
            if metric == "item_coverage_recs":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Render popularity-bias thesis figures")
    p.add_argument("--topk", required=True, help="topk-mode results CSV")
    p.add_argument("--rollout", required=True, help="rollout-mode results CSV")
    p.add_argument("--real", default=None, help="real-continuation reference CSV")
    p.add_argument("--out-dir", type=str, default=str(THESIS_DIR))
    args = p.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    frames = [load(args.topk, "topk"), load(args.rollout, "rollout")]
    if args.real:
        frames.append(load(args.real, "real"))
    stats = agg(pd.concat(frames, ignore_index=True))

    out_dir = Path(args.out_dir)
    fig_dgap(stats, out_dir / "popbias_dgap")
    fig_diversity(stats, out_dir / "popbias_diversity")
    print(f"Wrote popbias_dgap.{{pdf,png}} and popbias_diversity.{{pdf,png}} to {out_dir}")


if __name__ == "__main__":
    main()
