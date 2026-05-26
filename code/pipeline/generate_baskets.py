"""
generate_baskets.py - Recommendation-basket dump for fairness eval

Loads a trained SessionTransformer (the one named in config.yaml's
eval_model), builds N unique user contexts, and writes per-user
recommendation baskets to parquet. Basket-generation defaults come from
basket-config.yaml; CLI flags override that file.

Default output:
    output/models/<EVAL_MODEL_NAME>/baskets/baskets-<DDMMYY-HH-MM-SS>.parquet

Usage:
    PYTHONPATH=. uv run python generate_baskets.py
    PYTHONPATH=. uv run python generate_baskets.py --n-users 1000 --k 20 --rec-action product_buy
    PYTHONPATH=. uv run python generate_baskets.py --mode rollout --actions product_buy,add_to_cart
    PYTHONPATH=. uv run python generate_baskets.py --config basket-config.yaml

Top-K mode requires rec_action from basket-config.yaml or --rec-action, and
accepts positive item actions only:
    add_to_cart or product_buy

Rollout mode ignores --rec-action. Use --actions there to choose which sampled
event types become basket items.

Round-trip to dict
    from evaluation.baskets import load_as_dict
    d = load_as_dict("output/.../baskets-xxx.parquet")
    # {user_id: [item_id, ...]}
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml

from config import EVAL_MODEL_NAME, EVAL_MODEL_SUBDIR, MODEL_DIR
from evaluation.baskets import (
    DEFAULT_BASKET_ACTIONS,
    DEFAULT_INPUT_SOURCE,
    generate_baskets,
    save,
)


BASKET_CONFIG_PATH = Path(__file__).parent / "basket-config.yaml"
POSITIVE_REC_ACTIONS = ("add_to_cart", "product_buy")


def _require_eval_model() -> None:
    if EVAL_MODEL_SUBDIR is None:
        raise RuntimeError(
            "config.yaml must set eval_model to the folder name of the "
            "trained model to evaluate."
        )
    if not EVAL_MODEL_SUBDIR.exists():
        raise FileNotFoundError(f"eval_model folder not found: {EVAL_MODEL_SUBDIR}")


def _load_basket_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"basket config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"basket config must be a YAML mapping: {path}")
    return cfg


def _cfg_default(cfg: dict, key: str, fallback=None):
    return cfg[key] if key in cfg else fallback


def _actions_default(cfg: dict) -> str:
    actions = _cfg_default(cfg, "actions", DEFAULT_BASKET_ACTIONS)
    if isinstance(actions, str):
        return actions
    return ",".join(actions)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(BASKET_CONFIG_PATH),
                     help="Basket-generation YAML config path")
    pre_args, _ = pre.parse_known_args()
    cfg_path = Path(pre_args.config)
    cfg = _load_basket_config(cfg_path)

    p = argparse.ArgumentParser(
        description="Generate recommendation baskets for fairness eval",
        parents=[pre],
    )
    p.add_argument("--n-users",    type=int, default=_cfg_default(cfg, "n_users", 1000),
                   help="Number of users / baskets")
    p.add_argument("--k",          type=int, default=_cfg_default(cfg, "k", 20),
                   help="Basket size")
    p.add_argument("--seed",       type=int, default=_cfg_default(cfg, "seed", 42))
    p.add_argument("--mode",       type=str, default=_cfg_default(cfg, "mode", "topk"), choices=["topk", "rollout"],
                   help="topk: take the k most probable SKUs from the item head "
                        "(exactly k, default). rollout: sample a session and filter "
                        "events (variable-size, often << k).")
    p.add_argument("--input-source", type=str, default=_cfg_default(cfg, "input_source", DEFAULT_INPUT_SOURCE),
                   choices=["train", "sampler"],
                   help="train: unique real train users with prior-session history "
                        "(default, best for recommendation baskets). sampler: legacy "
                        "identity sampler/random fallback with unique user IDs.")
    p.add_argument("--rec-action", type=str, default=_cfg_default(cfg, "rec_action", None),
                   choices=POSITIVE_REC_ACTIONS,
                   help="Required in topk mode unless set by config. Action the item head "
                        "is conditioned on (for example: add_to_cart or product_buy). "
                        "Ignored in rollout mode.")
    p.add_argument("--top-c",      type=int, default=_cfg_default(cfg, "top_c", 100),
                   help="Candidate categories scored per user in topk hier/svdpq mode")
    p.add_argument("--actions",    type=str, default=_actions_default(cfg),
                   help=f"[rollout mode] Comma-separated basket-defining action names "
                        f"(default: {','.join(DEFAULT_BASKET_ACTIONS)})")
    p.add_argument("--batch-size", type=int, default=_cfg_default(cfg, "batch_size", 512))
    p.add_argument("--algorithm",  type=str, default=_cfg_default(cfg, "algorithm", None),
                   help="Algorithm tag written to the output (default: eval_model folder name)")
    p.add_argument("--out",        type=str, default=_cfg_default(cfg, "out", None),
                   help="Output parquet path (default: <eval_model>/baskets/baskets-<stamp>.parquet)")
    args = p.parse_args()
    args.config_path = cfg_path
    if args.mode == "topk" and not args.rec_action:
        p.error("rec_action is required when --mode topk; set it in basket-config.yaml or pass --rec-action")
    return args


def main():
    args = parse_args()
    _require_eval_model()

    t0 = time.time()
    model_path = EVAL_MODEL_SUBDIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    sampler_path = MODEL_DIR / "identity_sampler.pkl"

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%d%m%y-%H-%M-%S")
        out_path = EVAL_MODEL_SUBDIR / "baskets" / f"baskets-{stamp}.parquet"

    basket_actions = tuple(a.strip() for a in args.actions.split(",") if a.strip())

    if args.mode == "topk":
        print(f"\nGenerating {args.n_users:,} top-K baskets "
              f"(k={args.k}, rec_action={args.rec_action}, top_c={args.top_c}, "
              f"input_source={args.input_source}) ...")
    else:
        print(f"\nGenerating {args.n_users:,} rollout baskets "
              f"(k={args.k}, actions={basket_actions}, input_source={args.input_source}) ...")
    print(f"  config          : {args.config_path}")
    df = generate_baskets(
        model_path            = model_path,
        n_users               = args.n_users,
        k                     = args.k,
        seed                  = args.seed,
        mode                  = args.mode,
        input_source          = args.input_source,
        rec_action            = args.rec_action,
        top_c                 = args.top_c,
        basket_actions        = basket_actions,
        batch_size            = args.batch_size,
        identity_sampler_path = sampler_path if sampler_path.exists() else None,
        algorithm             = args.algorithm,
    )

    save(df, out_path)

    n_with_basket = df.attrs.get("n_users_with_basket", 0)
    sizes = df[df["rank"] >= 0].groupby("user_id")["rank"].count()
    mean_size = float(sizes.mean()) if len(sizes) else 0.0
    print(
        f"\nDone in {time.time() - t0:.1f}s\n"
        f"  config          : {args.config_path}\n"
        f"  mode            : {args.mode}\n"
        f"  input source    : {args.input_source}\n"
        f"  users requested : {args.n_users:,}\n"
        f"  users w/ basket : {n_with_basket:,}\n"
        f"  empty baskets   : {df.attrs.get('empty_baskets', 0):,}\n"
        f"  short baskets   : {df.attrs.get('short_baskets', 0):,} (< k items)\n"
        f"  mean basket size: {mean_size:.2f}\n"
        f"  output          : {out_path}"
    )


if __name__ == "__main__":
    main()
