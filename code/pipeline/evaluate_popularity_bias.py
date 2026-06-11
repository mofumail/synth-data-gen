"""
evaluate_popularity_bias.py - Popularity-bias evaluation of recommendation
baskets

Ingests one or more basket parquet files (written by generate_baskets.py)
and scores each through the four popularity-bias metrics of Braun et al.
(2023, arXiv:2310.08455): within-group Gini coefficient, dynamic-dGAP,
Between-group GAP, and group cosine similarity. See
evaluation/popularity_bias.py for the metric definitions and how the
paper's sensitive user groups are adapted to this dataset.

Results are written as both a long-form CSV (label, basket_file, group,
metric, value) for matplotlib/seaborn and a nested JSON that also records
the run configuration.

Usage (one basket file per model iteration, labelled in order):
    PYTHONPATH=. uv run python evaluate_popularity_bias.py \
        --baskets output/models/<iter1>/baskets/baskets-x.parquet \
                  output/models/<iter2>/baskets/baskets-y.parquet \
                  output/models/<iter3>/baskets/baskets-z.parquet \
        --labels iter1 iter2 iter3

    # Generate a fresh basket for config.yaml's eval_model first, then score it:
    PYTHONPATH=. uv run python evaluate_popularity_bias.py --generate \
        --generate-args "--n-users 1000 --k 20"

    # Alternative grouping: user activity terciles
    PYTHONPATH=. uv run python evaluate_popularity_bias.py --baskets ... \
        --group-by activity --n-groups 3

Default output:
    output/popularity_bias/popularity_bias-<DDMMYY-HH-MM-SS>.{csv,json}
"""

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import EVAL_MODEL_NAME, EVAL_MODEL_SUBDIR, OUTPUT_DIR
from evaluation.popularity_bias import (
    GROUP_STRATEGIES,
    assign_groups,
    evaluate_baskets,
    item_popularity,
    load_basket_items,
    profile_stats,
    read_train_interactions,
    results_to_long_df,
)


DEFAULT_OUT_DIR = OUTPUT_DIR / "popularity_bias"


def parse_args():
    p = argparse.ArgumentParser(
        description="Score recommendation baskets on the four popularity-bias "
                    "metrics of Braun et al. (2023)."
    )
    p.add_argument("--baskets", type=str, nargs="*", default=[],
                   help="Basket parquet path(s) from generate_baskets.py, one per model")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Plot label per basket file (default: model folder name). "
                        "Must match --baskets in count and order.")
    p.add_argument("--generate", action="store_true",
                   help="Run generate_baskets.py for config.yaml's eval_model first "
                        "and include the fresh basket in the evaluation")
    p.add_argument("--generate-args", type=str, default="",
                   help='Extra CLI flags forwarded to generate_baskets.py, e.g. '
                        '"--n-users 1000 --k 20"')
    p.add_argument("--group-by", type=str, default="profile_pop", choices=GROUP_STRATEGIES,
                   help="User grouping: profile_pop = mean popularity of train-profile "
                        "items (niche vs mainstream), activity = train profile size")
    p.add_argument("--n-groups", type=int, default=2,
                   help="Number of quantile groups (default 2)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Directory for the CSV/JSON results")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.baskets):
        p.error(f"--labels count ({len(args.labels)}) must match "
                f"--baskets count ({len(args.baskets)})")
    if not args.baskets and not args.generate:
        p.error("nothing to evaluate: pass --baskets and/or --generate")
    if args.n_groups < 2:
        p.error("--n-groups must be >= 2 (between-group metrics need pairs)")
    return args


def _run_generate(extra_args: str) -> Path:
    """Invoke generate_baskets.py for the configured eval_model; return the
    parquet path it wrote (pinned via --out so it is known up front)."""
    if EVAL_MODEL_SUBDIR is None:
        raise RuntimeError("--generate requires eval_model to be set in config.yaml")
    stamp = datetime.now().strftime("%d%m%y-%H-%M-%S")
    out_path = EVAL_MODEL_SUBDIR / "baskets" / f"baskets-{stamp}.parquet"
    cmd = [sys.executable, "generate_baskets.py", "--out", str(out_path)]
    cmd += shlex.split(extra_args)
    print(f"Generating baskets for eval_model={EVAL_MODEL_NAME} ...\n  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
    return out_path


def _default_label(path: Path, meta: dict) -> str:
    # baskets live at output/models/<model>/baskets/<file>.parquet; the model
    # folder name is the most readable identifier, falling back to the
    # algorithm tag stored in the file.
    if path.parent.name == "baskets":
        return path.parent.parent.name
    return meta["algorithm"]


def _unique(label: str, taken: set[str]) -> str:
    if label not in taken:
        return label
    i = 2
    while f"{label}#{i}" in taken:
        i += 1
    return f"{label}#{i}"


def _fmt(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _print_summary(label: str, metrics: dict) -> None:
    print(f"\n  {label}  "
          f"(users evaluated: {metrics['n_users_evaluated']:,}, "
          f"unmatched: {metrics['n_users_unmatched']:,})")
    for g, vals in metrics["groups"].items():
        print(f"    [{g}] n={vals['n_users']:,}  "
              f"GAP_p={_fmt(vals['gap_p'])}  GAP_r={_fmt(vals['gap_r'])}  "
              f"dGAP={_fmt(vals['delta_gap'])}  "
              f"gini(prof+recs)={_fmt(vals['gini_profiles_plus_recs'])}  "
              f"gini(recs)={_fmt(vals['gini_recs'])}  "
              f"coverage={_fmt(vals['item_coverage_recs'])}")
    for pair, vals in metrics["pairs"].items():
        print(f"    [{pair}] between-group GAP={_fmt(vals['between_group_gap'])}  "
              f"cosine={_fmt(vals['cosine_similarity'])}")


def main():
    args = parse_args()

    basket_paths = [Path(b) for b in args.baskets]
    labels = list(args.labels) if args.labels else [None] * len(basket_paths)
    if args.generate:
        basket_paths.append(_run_generate(args.generate_args))
        labels.append(None)
    for path in basket_paths:
        if not path.exists():
            raise FileNotFoundError(f"basket parquet not found: {path}")

    print("Loading train interactions (popularity scores, profiles, groups) ...")
    interactions = read_train_interactions()
    phi = item_popularity(interactions)
    stats = profile_stats(interactions, phi)
    groups = assign_groups(stats, strategy=args.group_by, n_groups=args.n_groups)
    print(f"  train users: {len(stats):,}  catalog items: {phi.size:,}  "
          f"groups ({args.group_by}): "
          f"{', '.join(f'{g}={n:,}' for g, n in groups.value_counts().items())}")

    results: dict[str, dict] = {}
    for path, label in zip(basket_paths, labels):
        pairs, meta = load_basket_items(path)
        label = _unique(label or _default_label(path, meta), set(results))
        print(f"\nEvaluating {path.name} as '{label}' "
              f"({meta['n_users_with_basket']:,} users, "
              f"mean basket size {meta['mean_basket_size']:.1f}, mode={meta['mode']})")
        metrics = evaluate_baskets(pairs, interactions, phi, groups)
        if metrics["n_users_unmatched"]:
            print(f"  [warn] {metrics['n_users_unmatched']:,} basket users have no "
                  f"train profile (sampler-generated IDs?) and were skipped")
        results[label] = {**meta, "metrics": metrics}
        _print_summary(label, metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%y-%H-%M-%S")
    csv_path = out_dir / f"popularity_bias-{stamp}.csv"
    json_path = out_dir / f"popularity_bias-{stamp}.json"

    results_to_long_df(results).to_csv(csv_path, index=False)
    payload = {
        "config": {
            "group_by": args.group_by,
            "n_groups": args.n_groups,
            "n_train_users": int(len(stats)),
            "n_catalog_items": int(phi.size),
            "group_sizes": {g: int(n) for g, n in groups.value_counts().items()},
            "paper": "Braun, Bhaumik & Dey (2023), arXiv:2310.08455",
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float))

    print(f"\nDone.\n  csv  : {csv_path}\n  json : {json_path}")


if __name__ == "__main__":
    main()
